from pathlib import Path

import psycopg
from dotenv import dotenv_values
from services.google_auth import get_google_services


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG = dotenv_values(BASE_DIR / ".env")

DATABASE_URL = CONFIG.get("DATABASE_URL")


def get_db_connection():
    """
    Create a PostgreSQL connection using the project's .env settings.
    """

    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is missing from the project .env file."
        )

    # Convert SQLAlchemy-style URL to a standard psycopg URL.
    connection_url = DATABASE_URL.replace(
        "postgresql+psycopg://",
        "postgresql://",
        1,
    )

    return psycopg.connect(connection_url)


def test_database_connection():
    """
    Verify database connectivity and the processing ledger table.
    """

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()

            cursor.execute("""
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'email_processing'
                )
            """)
            table_exists = cursor.fetchone()[0]

    if result[0] != 1:
        raise RuntimeError("Database connection test failed.")

    if not table_exists:
        raise RuntimeError("email_processing table was not found.")

    return True


def claim_email(message_id, thread_id=None, received_at=None):
    """
    Atomically claim an email for processing.

    Returns True if this execution claimed it.
    Returns False if the message already exists.
    """

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO email_processing (
                    message_id,
                    thread_id,
                    received_at,
                    processing_status,
                    current_step
                )
                VALUES (%s, %s, %s, 'PROCESSING', 'claimed')

                ON CONFLICT (message_id) DO UPDATE
                SET
                    thread_id = EXCLUDED.thread_id,
                    received_at = EXCLUDED.received_at,
                    processing_status = 'PROCESSING',
                    current_step = 'claimed',
                    error_message = NULL,
                    retry_count = email_processing.retry_count + 1,
                    updated_at = NOW()

                WHERE
                    email_processing.processing_status = 'PENDING'
                    OR (
                    email_processing.processing_status = 'PROCESSING'
                    AND email_processing.current_step = 'claimed'
                    AND email_processing.updated_at < NOW() - INTERVAL '15 minutes')

                RETURNING message_id
                """,
                (message_id, thread_id, received_at),
            )

            claimed = cursor.fetchone() is not None

    return claimed


def update_email_status(
    message_id,
    processing_status,
    current_step=None,
    category=None,
    confidence=None,
    error_message=None,
):
    """
    Update the processing record for an email.
    """

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE email_processing
                SET
                    processing_status = %s,
                    current_step = COALESCE(%s, current_step),
                    category = COALESCE(%s, category),
                    confidence = COALESCE(%s, confidence),
                    error_message = %s,
                    updated_at = NOW(),
                    completed_at = CASE
                        WHEN %s IN ('COMPLETED', 'SKIPPED', 'REVIEW')
                        THEN NOW()
                        ELSE completed_at
                    END
                WHERE message_id = %s
                """,
                (
                    processing_status,
                    current_step,
                    category,
                    confidence,
                    error_message,
                    processing_status,
                    message_id,
                ),
            )

def check_attachments_downloaded(message_id):

    gmail, _drive = get_google_services()


    