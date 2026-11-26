"""
Download supported attachments from emails returned by email_intake.py.

Original attachment filenames are preserved. To prevent two different emails
from overwriting files with the same name, each email gets its own local
message-ID subfolder:

data/attachments/staging/<gmail-message-id>/<original-attachment-name>

A small metadata JSON file records the Gmail message ID and attachment index.
No email is modified and no Google Drive upload is performed here.
"""

import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable

from services.google_auth import get_google_services
from email_intake import fetch_eligible_inbox_messages


PROJECT_DIR = Path(__file__).resolve().parent
STAGING_DIR = PROJECT_DIR / "data" / "attachments" / "staging"

SUPPORTED_MIME_PREFIXES = ("image/",)
SUPPORTED_MIME_TYPES = {
    "application/pdf",
    "application/x-pdf",
}


def walk_mime_parts(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield MIME parts recursively."""
    if not payload:
        return

    yield payload
    for part in payload.get("parts", []) or []:
        yield from walk_mime_parts(part)


def safe_filename(filename: str) -> str:
    """Keep the sender's filename while removing Windows-invalid characters."""
    name = Path(filename.replace("\\", "/")).name
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = name.strip(" .")
    return name or "attachment"


def decode_gmail_data(data: str) -> bytes:
    """Decode Gmail's URL-safe base64 attachment data."""
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def download_attachment_bytes(
    gmail,
    message_id: str,
    part: Dict[str, Any],
) -> bytes:
    """Download one MIME attachment and return its bytes."""
    body = part.get("body", {}) or {}
    attachment_id = body.get("attachmentId")

    if attachment_id:
        response = (
            gmail.users()
            .messages()
            .attachments()
            .get(
                userId="me",
                messageId=message_id,
                id=attachment_id,
            )
            .execute()
        )
        encoded_data = response.get("data", "")
    else:
        encoded_data = body.get("data", "")

    if not encoded_data:
        raise ValueError("Gmail returned no attachment data.")

    return decode_gmail_data(encoded_data)


def ensure_correct_extension(filename: str, file_bytes: bytes, mime_type: str) -> str:
    """Keep the original base filename and ensure actual PDFs end in .pdf."""
    path = Path(filename)
    is_pdf = file_bytes.startswith(b"%PDF-") or mime_type in SUPPORTED_MIME_TYPES

    if is_pdf and path.suffix.lower() != ".pdf":
        return f"{path.stem}.pdf"

    return filename


def attachment_path(directory: Path, filename: str, occurrence: int) -> Path:
    """Return a stable path; only same-name files within one email get a suffix."""
    if occurrence <= 1:
        return directory / filename

    path = Path(filename)
    return directory / f"{path.stem} ({occurrence}){path.suffix}"


def main() -> None:
    eligible_messages = fetch_eligible_inbox_messages()

    if not eligible_messages:
        print("\nNo eligible emails were found. No attachments were downloaded.")
        return

    gmail, _drive = get_google_services()
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    downloaded_count = 0
    skipped_count = 0
    failed_count = 0

    print(f"\nStaging folder: {STAGING_DIR}")

    for email_item in eligible_messages:
        message_id = email_item["message_id"]
        message_dir = STAGING_DIR / message_id
        message_dir.mkdir(parents=True, exist_ok=True)

        try:
            full_message = (
                gmail.users()
                .messages()
                .get(userId="me", id=message_id, format="full")
                .execute()
            )
        except Exception as exc:
            failed_count += 1
            print(
                f"[ERROR] Could not retrieve message {message_id[:12]}: "
                f"{type(exc).__name__}"
            )
            continue

        payload = full_message.get("payload", {}) or {}
        attachment_number = 0
        filename_occurrences = {}
        message_metadata = []

        for part in walk_mime_parts(payload):
            original_name = part.get("filename") or ""
            if not original_name:
                continue

            mime_type = (part.get("mimeType") or "").lower()
            is_supported = (
                mime_type in SUPPORTED_MIME_TYPES
                or any(mime_type.startswith(prefix)
                       for prefix in SUPPORTED_MIME_PREFIXES)
            )

            if not is_supported:
                skipped_count += 1
                print(
                    f"[SKIP] Unsupported attachment: "
                    f"{safe_filename(original_name)} ({mime_type})"
                )
                continue

            attachment_number += 1
            clean_name = safe_filename(original_name)

            try:
                file_bytes = download_attachment_bytes(gmail, message_id, part)
                final_name = ensure_correct_extension(
                    clean_name, file_bytes, mime_type
                )
                filename_key = final_name.casefold()
                filename_occurrences[filename_key] = (
                    filename_occurrences.get(filename_key, 0) + 1
                )
                destination = attachment_path(
                    message_dir,
                    final_name,
                    filename_occurrences[filename_key],
                )
                destination.write_bytes(file_bytes)

                body = part.get("body", {}) or {}
                message_metadata.append({
                    "message_id": message_id,
                    "attachment_index": attachment_number,
                    "attachment_id": body.get("attachmentId"),
                    "original_filename": original_name,
                    "saved_filename": destination.name,
                    "mime_type": mime_type,
                    "byte_count": len(file_bytes),
                })

                downloaded_count += 1
                print(
                    f"[DOWNLOADED] {destination.name} "
                    f"({len(file_bytes):,} bytes)"
                )

            except Exception as exc:
                failed_count += 1
                print(
                    f"[ERROR] Could not download {clean_name} "
                    f"from message {message_id[:12]}: {type(exc).__name__}"
                )

        metadata_path = message_dir / "_attachment_metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "message_id": message_id,
                    "attachments": message_metadata,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print("\nAttachment staging completed.")
    print(f"Downloaded PDF/image attachments: {downloaded_count}")
    print(f"Skipped unsupported attachments: {skipped_count}")
    print(f"Failed downloads: {failed_count}")
    print(f"Local staging folder: {STAGING_DIR}")
    print("Original attachment filenames are preserved.")
    print("No email was modified and nothing was uploaded to Google Drive.")


if __name__ == "__main__":
    main()

