
import json
from pathlib import Path

from services.embedding_service import generate_embedding


BASE_DIR = Path(__file__).resolve().parent

JOB_ID = "DS-001"

ATS_JSON_PATH = (
    BASE_DIR / "data" / "resumes" / "ats" / f"{JOB_ID}.json"
)

OUTPUT_PATH = (
    BASE_DIR / "data" / "resumes" / "ats"
    / f"{JOB_ID}_embeddings.json"
)


def build_skills_text(profile):
    required = profile.get("required_skills", [])
    preferred = profile.get("preferred_skills", [])

    return (
        "Required skills:\n"
        + "\n".join(required)
        + "\nPreferred skills:\n"
        + "\n".join(preferred)
    )


def build_projects_text(profile):
    projects = profile.get("required_projects", [])

    return "Expected projects:\n" + "\n".join(projects)


def build_whole_profile_text(profile):
    return json.dumps(
        profile,
        indent=2,
        ensure_ascii=False
    )


def main():
    if not ATS_JSON_PATH.is_file():
        raise FileNotFoundError(
            f"ATS JSON file not found: {ATS_JSON_PATH}"
        )

    print(f"Loading ATS profile: {JOB_ID}")

    with open(ATS_JSON_PATH, "r", encoding="utf-8") as file:
        profile = json.load(file)

    # Validate required sections
    required_fields = [
        "required_skills",
        "preferred_skills",
        "required_projects",
        "required_experience"
    ]

    for field in required_fields:
        if field not in profile:
            raise ValueError(f"Missing ATS field: {field}")

    experience = profile["required_experience"]

    if not isinstance(experience, dict):
        raise ValueError("required_experience must be an object.")

    print("\nGenerating whole profile embedding...")
    whole_embedding = generate_embedding(
        build_whole_profile_text(profile)
    )

    print("Generating skills embedding...")
    skills_embedding = generate_embedding(
        build_skills_text(profile)
    )

    print("Generating projects embedding...")
    project_text = build_projects_text(profile)

    if not profile["required_projects"]:
        raise ValueError("No required projects found in ATS JSON.")

    project_embedding = generate_embedding(project_text)

    # Experience is compared numerically in months.
    # No experience embedding is generated.
    output = {
        "job_id": JOB_ID,
        "whole_embedding": whole_embedding,
        "skills_embedding": skills_embedding,
        "project_embedding": project_embedding,
        "experience_embedding": None,
        "minimum_months": experience["minimum_months"],
        "preferred_months": experience["preferred_months"],
        "embedding_model": "nomic-embed-text",
        "dimensions": 768
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print("\nATS embeddings generated successfully!")
    print(f"Job ID: {JOB_ID}")
    print(f"Whole embedding dimensions: {len(whole_embedding)}")
    print(f"Skills embedding dimensions: {len(skills_embedding)}")
    print(f"Project embedding dimensions: {len(project_embedding)}")
    print("Experience embedding: Not generated (numeric comparison)")
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
