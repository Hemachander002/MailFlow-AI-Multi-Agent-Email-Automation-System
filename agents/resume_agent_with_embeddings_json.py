"""
Resume Agent orchestrator.

Uses the existing working services:
- services.resume_parser.extract_resume_text
- services.gemini_resume_parser.parse_resume_with_gemini
- services.embedding_service.generate_embedding

The candidate/ATS embedding text construction follows the supplied
generate_candidate_embeddings.py and generate_ats_embeddings.py scripts.

Run from the project root:
    python agents/resume_agent.py --pdf "data/resumes/candidate/Candidate_resume.pdf" --job-id DS-001

Call from another module:
    from agents.resume_agent import process_resume
    result = process_resume(pdf_path, job_id)
"""

import json
import logging
import math
import re
import sys
import time
from pathlib import Path

import psycopg2
from dotenv import dotenv_values
from pgvector.psycopg2 import register_vector
from psycopg2.extras import Json

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from services.resume_parser import extract_resume_text
from services.gemini_resume_parser import parse_resume_with_gemini
from services.embedding_service import generate_embedding
from app_observability import observe_api_call, observe_agent, record_resume_success


# ---------------------------------------------------------------------
# Configuration and logging
# ---------------------------------------------------------------------

CONFIG = dotenv_values(BASE_DIR / ".env")
DATABASE_URL = CONFIG.get("DATABASE_URL")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("resume_agent")


# ---------------------------------------------------------------------
# Database connection - same connection approach as the supplied,
# successfully tested test_db.py / insert_candidate.py / insert_ats.py.
# ---------------------------------------------------------------------

def get_db_connection():
    """Connect to PostgreSQL and verify the connection with SELECT 1."""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL missing from .env")

    database_url = DATABASE_URL

    if database_url.startswith("postgresql+psycopg://"):
        database_url = database_url.replace(
            "postgresql+psycopg://",
            "postgresql://",
            1
        )

    print("\n[DATABASE] Connecting to PostgreSQL...", flush=True)

    try:
        connection = psycopg2.connect(
            database_url,
            host="127.0.0.1",
            connect_timeout=10
        )
        register_vector(connection)

        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT 1, current_database(), current_user, "
                "inet_server_addr()::text, inet_server_port()"
            )
            check_value, database_name, database_user, server_address, server_port = cursor.fetchone()

        if check_value != 1:
            connection.close()
            raise RuntimeError("PostgreSQL connection test did not return 1.")

        print("[DATABASE] SUCCESS: PostgreSQL connection established.", flush=True)
        print(f"[DATABASE] Database: {database_name}", flush=True)
        print(f"[DATABASE] User: {database_user}", flush=True)
        print(f"[DATABASE] Server: {server_address}:{server_port}", flush=True)
        return connection

    except Exception as exc:
        print(f"[DATABASE] FAILED: {type(exc).__name__}", flush=True)
        raise


# ---------------------------------------------------------------------
# Gemini call with retry for temporary server/rate-limit errors.
# The existing Gemini parser and its prompt/schema are not modified.
# ---------------------------------------------------------------------

def parse_resume_with_retry(resume_text, attempts=4):
    retryable_codes = ("429", "500", "502", "503", "504")

    for attempt in range(1, attempts + 1):
        try:
            with observe_api_call("gemini", "resume_parsing"):
                return parse_resume_with_gemini(resume_text)
        except Exception as exc:
            message = str(exc)
            code = str(
                getattr(exc, "code", "")
                or getattr(exc, "status_code", "")
            )

            response = getattr(exc, "response", None)
            if response is not None:
                code = code or str(getattr(response, "status_code", ""))

            is_retryable = (
                any(code_part in code for code_part in retryable_codes)
                or any(code_part in message for code_part in retryable_codes)
            )

            if not is_retryable or attempt == attempts:
                raise

            wait_seconds = 2 ** attempt
            logger.warning(
                "Gemini temporarily returned an error. "
                "Retry %s/%s in %s seconds.",
                attempt,
                attempts - 1,
                wait_seconds
            )
            time.sleep(wait_seconds)


# ---------------------------------------------------------------------
# Candidate text construction - same strategy as the supplied
# generate_candidate_embeddings.py.
# ---------------------------------------------------------------------

def build_candidate_skills_text(profile):
    skills = profile.get("skills") or []
    skills = [str(item).strip() for item in skills if str(item).strip()]

    if not skills:
        return None

    return "Candidate skills:\n" + "\n".join(skills)


def build_candidate_projects_text(profile):
    projects = profile.get("projects") or []
    projects = [str(item).strip() for item in projects if str(item).strip()]

    if not projects:
        return None

    return "Candidate projects:\n" + "\n".join(projects)


def generate_candidate_embeddings(resume_text, profile):
    if not resume_text or not resume_text.strip():
        raise ValueError("Original resume text is empty.")

    # Whole resume: original extracted PDF text.
    logger.info("Generating whole-resume embedding...")
    whole_embedding = generate_embedding(resume_text)

    # Skills: Gemini-extracted skills, using the existing text prefix.
    skills_text = build_candidate_skills_text(profile)
    if skills_text:
        logger.info("Generating candidate skills embedding...")
        skills_embedding = generate_embedding(skills_text)
    else:
        logger.warning("Candidate skills are missing; skills score will be 0.")
        skills_embedding = None

    # Projects: Gemini-extracted projects, using the existing text prefix.
    projects_text = build_candidate_projects_text(profile)
    if projects_text:
        logger.info("Generating candidate projects embedding...")
        project_embedding = generate_embedding(projects_text)
    else:
        logger.warning("Candidate projects are missing; project score will be 0.")
        project_embedding = None

    return {
        "whole_embedding": whole_embedding,
        "skills_embedding": skills_embedding,
        "project_embedding": project_embedding,
        # Experience is numeric; do not create an experience embedding.
        "experience_embedding": None,
        "experience_months": parse_experience_months(
            profile.get("total_experience_years")
        )
    }


def save_candidate_embeddings_json(candidate_embeddings, profile, job_id):
    """
    Save candidate embeddings locally for debugging.

    The JSON layout follows the existing generate_candidate_embeddings.py
    output format. Missing skills/projects remain JSON null; no empty
    embeddings are generated.
    """
    output_dir = BASE_DIR / "data" / "resumes" / "candidate"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "candidate_embeddings.json"

    output = {
        "whole_embedding": candidate_embeddings["whole_embedding"],
        "skills_embedding": candidate_embeddings["skills_embedding"],
        "project_embedding": candidate_embeddings["project_embedding"],
        "experience_embedding": candidate_embeddings["experience_embedding"],
        "total_experience_years": profile.get("total_experience_years"),
        "experience_months": candidate_embeddings["experience_months"],
        "embedding_model": "nomic-embed-text",
        "dimensions": 768,
        "job_id": job_id
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"[EMBEDDINGS] Candidate embeddings JSON saved to: {output_path}", flush=True)

    for field in ("whole_embedding", "skills_embedding", "project_embedding"):
        vector = output[field]
        if vector is None:
            print(f"[EMBEDDINGS] {field}: not generated (missing source data)", flush=True)
        else:
            print(f"[EMBEDDINGS] {field}: {len(vector)} dimensions", flush=True)

    return output_path


# ---------------------------------------------------------------------
# Experience conversion.
# Gemini's schema names this field total_experience_years.
# A bare numeric value is therefore interpreted as years.
# ---------------------------------------------------------------------

def parse_experience_months(value):
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)) or float(value) < 0:
            return None
        return round(float(value) * 12)

    text = str(value).strip().lower()
    if not text:
        return None

    if text in {
        "fresher",
        "fresh graduate",
        "no experience",
        "0",
        "0.0",
        "0 year",
        "0 years",
        "0 month",
        "0 months"
    }:
        return 0

    year_match = re.search(r"(\d+(?:\.\d+)?)\s*years?", text)
    month_match = re.search(r"(\d+(?:\.\d+)?)\s*months?", text)

    if year_match or month_match:
        years = float(year_match.group(1)) if year_match else 0.0
        months = float(month_match.group(1)) if month_match else 0.0

        if years < 0 or months < 0:
            return None

        return round(years * 12 + months)

    # Since the source field is total_experience_years, a bare number
    # such as "1.5" means 1.5 years.
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return round(float(text) * 12)

    return None


# ---------------------------------------------------------------------
# ATS retrieval.
# ATS vectors were generated from the ATS JSON using the supplied
# generate_ats_embeddings.py strategy and inserted into the ats table.
# ---------------------------------------------------------------------

def get_ats_profile(connection, job_id):
    """
    Retrieve ATS data. Cast pgvector columns to text in SQL so the
    driver returns ordinary strings rather than pgvector Vector objects.
    PostgreSQL pgvector text output is JSON-compatible: [0.1,0.2,...].
    """
    query = """
        SELECT
            job_id,
            whole_embedding::text,
            skills_embedding::text,
            project_embedding::text,
            fresher,
            minimum_months,
            preferred_months
        FROM ats
        WHERE job_id = %s
    """

    with connection.cursor() as cursor:
        cursor.execute(query, (job_id,))
        row = cursor.fetchone()

    if row is None:
        raise ValueError(f"No ATS row found for job_id: {job_id}")

    ats = {
        "job_id": row[0],
        "whole_embedding": json.loads(row[1]) if row[1] is not None else None,
        "skills_embedding": json.loads(row[2]) if row[2] is not None else None,
        "project_embedding": json.loads(row[3]) if row[3] is not None else None,
        "fresher": row[4],
        "minimum_months": row[5],
        "preferred_months": row[6]
    }

    for field in ("whole_embedding", "skills_embedding", "project_embedding"):
        if ats[field] is None:
            raise ValueError(
                f"ATS {field} is missing for job_id {job_id}. "
                "Check the ATS embedding generation/insertion step."
            )

        ats[field] = [float(value) for value in ats[field]]

        if len(ats[field]) != 768:
            raise ValueError(
                f"ATS {field} must contain 768 values; got {len(ats[field])}."
            )

    print(
        f"[DATABASE] ATS profile loaded for {job_id}. "
        "All three ATS vectors have 768 dimensions.",
        flush=True
    )
    return ats


# ---------------------------------------------------------------------
# Cosine similarity and agreed score conversion:
# score = clamp(cosine_similarity * 100, 0, 100)
# ---------------------------------------------------------------------

def cosine_similarity_score(candidate_vector, ats_vector):
    if candidate_vector is None:
        return 0.0

    if ats_vector is None:
        raise ValueError("ATS embedding is missing.")

    candidate_vector = list(candidate_vector)
    ats_vector = list(ats_vector)

    if len(candidate_vector) != len(ats_vector):
        raise ValueError("Candidate and ATS embedding dimensions differ.")

    dot_product = sum(
        a * b for a, b in zip(candidate_vector, ats_vector)
    )
    candidate_norm = math.sqrt(sum(a * a for a in candidate_vector))
    ats_norm = math.sqrt(sum(b * b for b in ats_vector))

    if candidate_norm == 0 or ats_norm == 0:
        return 0.0

    cosine = dot_product / (candidate_norm * ats_norm)
    cosine = max(-1.0, min(1.0, cosine))

    return round(max(0.0, min(100.0, cosine * 100.0)), 2)


# ---------------------------------------------------------------------
# Experience scoring: linear interpolation between minimum/preferred.
# ---------------------------------------------------------------------

def calculate_experience_score(
    candidate_months,
    minimum_months,
    preferred_months
):
    if candidate_months is None:
        return None

    minimum_months = int(minimum_months or 0)
    preferred_months = int(preferred_months or 0)

    if minimum_months < 0 or preferred_months < 0:
        raise ValueError("ATS experience months cannot be negative.")

    if preferred_months < minimum_months:
        raise ValueError(
            "ATS preferred_months is less than minimum_months."
        )

    if candidate_months < minimum_months:
        return 0.0

    if minimum_months == preferred_months:
        return 100.0

    if candidate_months >= preferred_months:
        return 100.0

    score = (
        (candidate_months - minimum_months)
        / (preferred_months - minimum_months)
    ) * 100.0

    return round(score, 2)


# ---------------------------------------------------------------------
# Matching weights:
# Fresher:     skills 87.5%, projects 12.5%, experience excluded.
# Non-fresher: skills 70%, projects 20%, experience 10%.
# Whole score is stored but has 0% weight.
# ---------------------------------------------------------------------

def calculate_matching_scores(candidate, ats):
    whole_score = cosine_similarity_score(
        candidate["whole_embedding"],
        ats["whole_embedding"]
    )

    skills_score = cosine_similarity_score(
        candidate["skills_embedding"],
        ats["skills_embedding"]
    )

    project_score = cosine_similarity_score(
        candidate["project_embedding"],
        ats["project_embedding"]
    )

    if ats["fresher"]:
        experience_score = None
        final_score = (
            skills_score * 0.875
            + project_score * 0.125
        )
        manual_review = False
    else:
        experience_score = calculate_experience_score(
            candidate["experience_months"],
            ats["minimum_months"],
            ats["preferred_months"]
        )

        if experience_score is None:
            # Unknown experience must not be silently treated as zero.
            final_score = None
            manual_review = True
        else:
            final_score = (
                skills_score * 0.70
                + project_score * 0.20
                + experience_score * 0.10
            )
            manual_review = False

    return {
        "whole_score": whole_score,
        "skills_score": skills_score,
        "project_score": project_score,
        "experience_score": experience_score,
        "final_score": (
            round(final_score, 2)
            if final_score is not None
            else None
        ),
        "manual_review": manual_review
    }


# ---------------------------------------------------------------------
# HR suggestion. No automated hiring/rejection decision is made.
# ---------------------------------------------------------------------

def generate_hr_suggestion(scores, ats, candidate):
    notes = []

    if scores["manual_review"]:
        notes.append(
            "MANUAL REVIEW REQUIRED: Candidate experience could not "
            "be reliably determined from the parsed resume."
        )

    if candidate["skills_embedding"] is None:
        notes.append("Candidate skills information is missing.")

    if candidate["project_embedding"] is None:
        notes.append("Candidate projects information is missing.")

    if ats["fresher"]:
        notes.append(
            "Fresher weights: skills 87.5%, projects 12.5%; "
            "experience excluded."
        )
    else:
        notes.append(
            "Non-fresher weights: skills 70%, projects 20%, "
            "experience 10%."
        )

    if scores["final_score"] is None:
        notes.append("Final weighted score is pending manual experience review.")
    else:
        notes.append(
            f"Final weighted matching score: "
            f"{scores['final_score']:.2f}/100."
        )

    notes.append(
        f"Whole-resume similarity: {scores['whole_score']:.2f}/100 "
        "(informational only; 0% final-score weight)."
    )
    notes.append(
        "This is an advisory matching result. HR must review the resume "
        "and make the final hiring decision."
    )

    return " ".join(notes)


# ---------------------------------------------------------------------
# Atomic persistence.
# Candidate insert/retrieval follows the supplied insert_candidate.py:
# ON CONFLICT(email_id) DO UPDATE SET email_id = EXCLUDED.email_id.
# HR application insert/update uses the existing unique(candidate_id, job_id).
# ---------------------------------------------------------------------

def save_candidate_and_application(
    connection,
    profile,
    pdf_path,
    job_id,
    candidate_embeddings,
    scores,
    suggestion
):
    email = profile.get("email")

    if not isinstance(email, str) or not email.strip() or "@" not in email:
        raise ValueError(
            "Candidate email is missing or invalid; cannot store application."
        )

    email = email.strip().lower()
    phone = profile.get("phone") or None

    candidate_query = """
        INSERT INTO candidate (
            email_id,
            phone_number,
            parsed_profile
        )
        VALUES (%s, %s, %s)
        ON CONFLICT (email_id)
        DO UPDATE SET
            email_id = EXCLUDED.email_id
        RETURNING candidate_id
    """

    hr_query = """
        INSERT INTO hr (
            candidate_id,
            job_id,
            resume_path,
            parsed_profile,
            whole_embedding,
            skills_embedding,
            project_embedding,
            experience_embedding,
            whole_score,
            skills_score,
            project_score,
            experience_score,
            suggestion
        )
        VALUES (
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s, %s
        )
        ON CONFLICT (candidate_id, job_id)
        DO UPDATE SET
            resume_path = EXCLUDED.resume_path,
            parsed_profile = EXCLUDED.parsed_profile,
            whole_embedding = EXCLUDED.whole_embedding,
            skills_embedding = EXCLUDED.skills_embedding,
            project_embedding = EXCLUDED.project_embedding,
            experience_embedding = EXCLUDED.experience_embedding,
            whole_score = EXCLUDED.whole_score,
            skills_score = EXCLUDED.skills_score,
            project_score = EXCLUDED.project_score,
            experience_score = EXCLUDED.experience_score,
            suggestion = EXCLUDED.suggestion
        RETURNING application_id
    """

    # psycopg2 connection context commits on success and rolls back
    # both inserts if either statement fails.
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(
                candidate_query,
                (email, phone, Json(profile))
            )
            candidate_id = cursor.fetchone()[0]

            cursor.execute(
                hr_query,
                (
                    candidate_id,
                    job_id,
                    str(pdf_path),
                    Json(profile),
                    candidate_embeddings["whole_embedding"],
                    candidate_embeddings["skills_embedding"],
                    candidate_embeddings["project_embedding"],
                    None,
                    scores["whole_score"],
                    scores["skills_score"],
                    scores["project_score"],
                    scores["experience_score"],
                    suggestion
                )
            )
            application_id = cursor.fetchone()[0]

    return {
        "candidate_id": str(candidate_id),
        "application_id": str(application_id)
    }


# ---------------------------------------------------------------------
# Main orchestration entry point.
# ---------------------------------------------------------------------

def _process_resume_impl(pdf_path, job_id):
    pdf_path = Path(pdf_path)
    if not pdf_path.is_absolute():
        pdf_path = BASE_DIR / pdf_path
    pdf_path = pdf_path.resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(f"Resume PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF resumes are supported.")

    if not job_id or not str(job_id).strip():
        raise ValueError("job_id is required.")

    job_id = str(job_id).strip()

    logger.info("Resume Agent started.")
    logger.info("Job ID: %s", job_id)

    # Connect and verify PostgreSQL before starting extraction/API work.
    connection = get_db_connection()

    try:
        # Retrieve ATS data early so missing job IDs/embeddings fail fast.
        ats = get_ats_profile(connection, job_id)
        logger.info("ATS record retrieved. Fresher role: %s", ats["fresher"])

        # 1. PDF -> extracted text (existing PyMuPDF/Tesseract strategy).
        print("\n[RESUME] Extracting text from PDF...", flush=True)
        resume_text = extract_resume_text(str(pdf_path))
        print(
            f"[RESUME] Text extraction successful ({len(resume_text)} characters).",
            flush=True
        )

        # 2. Extracted text -> Gemini candidate profile.
        print("[GEMINI] Parsing candidate resume...", flush=True)
        profile = parse_resume_with_retry(resume_text)

        if not isinstance(profile, dict):
            raise ValueError("Gemini returned an invalid candidate profile.")
        print("[GEMINI] Candidate profile parsed successfully.", flush=True)

        # 3. Candidate text/profile -> embeddings using existing Ollama service.
        candidate_embeddings = generate_candidate_embeddings(
            resume_text,
            profile
        )
        print("[EMBEDDINGS] Candidate embeddings generated.", flush=True)

        # Save the generated candidate vectors locally for debugging.
        save_candidate_embeddings_json(
            candidate_embeddings,
            profile,
            job_id
        )

        # 4. Calculate scores.
        scores = calculate_matching_scores(candidate_embeddings, ats)

        # 5. Build suggestion.
        suggestion = generate_hr_suggestion(
            scores,
            ats,
            candidate_embeddings
        )

        # 6. Save candidate + HR application atomically.
        print("[DATABASE] Saving candidate and HR application...", flush=True)
        saved = save_candidate_and_application(
            connection=connection,
            profile=profile,
            pdf_path=pdf_path,
            job_id=job_id,
            candidate_embeddings=candidate_embeddings,
            scores=scores,
            suggestion=suggestion
        )
        print("[DATABASE] SUCCESS: Candidate and HR application saved.", flush=True)

        logger.info("Resume Agent completed successfully.")
        return {
            "success": True,
            "job_id": job_id,
            **saved,
            "scores": scores,
            "suggestion": suggestion
        }

    except Exception as exc:
        connection.rollback()
        logger.error("Resume Agent failed. Error type: %s", type(exc).__name__)
        raise
    finally:
        connection.close()
        print("[DATABASE] Connection closed.", flush=True)


def process_resume(pdf_path, job_id):
    """Instrument the existing resume pipeline without changing its logic."""
    with observe_agent("resume"):
        result = _process_resume_impl(pdf_path, job_id)
        if isinstance(result, dict) and result.get("success"):
            record_resume_success()
        return result


# ---------------------------------------------------------------------
# USER INPUTS: edit these two values before running the script.
# Paths can be absolute or relative to the project root.
# ---------------------------------------------------------------------
PDF_PATH = r"data\resumes\candidate\Candidate_resume.pdf"
JOB_ID = "DS-001"


def main():
    print("========== RESUME AGENT ==========", flush=True)
    print(f"Job ID     : {JOB_ID}", flush=True)

    result = process_resume(PDF_PATH, JOB_ID)

    print("\n========== PROCESSING COMPLETED ==========", flush=True)
    print("Success        :", result["success"])
    print("Job ID         :", result["job_id"])
    print("Candidate application saved successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error("Resume processing failed. Error type: %s", type(exc).__name__)
        raise SystemExit(1)
