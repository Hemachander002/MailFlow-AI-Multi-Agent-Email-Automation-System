import json
from pathlib import Path

import psycopg2
from dotenv import dotenv_values
from pgvector.psycopg2 import register_vector

BASE_DIR = Path(__file__).resolve().parent
CONFIG = dotenv_values(BASE_DIR / ".env")

EMBEDDINGS_FILE = (
    BASE_DIR / "data" / "resumes" / "ats" / "DS-001_embeddings.json"
)

# Load generated ATS embeddings
with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)

# Validate embeddings before insertion
for field in ("whole_embedding", "skills_embedding", "project_embedding"):
    embedding = data.get(field)
    if not isinstance(embedding, list) or len(embedding) != 768:
        raise ValueError(f"Invalid or missing {field}")

database_url = CONFIG.get("DATABASE_URL")

if not database_url:
    raise ValueError("DATABASE_URL missing from .env")

if database_url.startswith("postgresql+psycopg://"):
    database_url = database_url.replace(
        "postgresql+psycopg://", "postgresql://", 1
    )

connection = None

try:
    connection = psycopg2.connect(
        database_url,
        host="127.0.0.1"
    )

    register_vector(connection)

    with connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO ats (
                    job_id,
                    whole_embedding,
                    skills_embedding,
                    project_embedding,
                    experience_embedding,
                    fresher,
                    minimum_months,
                    preferred_months,
                    updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())

                ON CONFLICT (job_id)
                DO UPDATE SET
                    whole_embedding = EXCLUDED.whole_embedding,
                    skills_embedding = EXCLUDED.skills_embedding,
                    project_embedding = EXCLUDED.project_embedding,
                    experience_embedding = EXCLUDED.experience_embedding,
                    fresher = EXCLUDED.fresher,
                    minimum_months = EXCLUDED.minimum_months,
                    preferred_months = EXCLUDED.preferred_months,
                    updated_at = NOW()
                """,
                (
                    data["job_id"],
                    data["whole_embedding"],
                    data["skills_embedding"],
                    data["project_embedding"],
                    None,
                    True,  # DS-001 is a fresher position
                    data["minimum_months"],
                    data["preferred_months"],
                ),
            )

    print("ATS profile inserted/updated successfully!")
    print("Job ID:", data["job_id"])
    print("Fresher:", True)

except Exception:
    print("ATS profile insertion failed.")
    raise

finally:
    if connection is not None:
        connection.close()
