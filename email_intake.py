"""
Read-only Gmail intake for the multi-agent email automation project.

This script:
- Reads messages in the Gmail Inbox received strictly AFTER
  13:00 Asia/Kolkata on 24 November 2026.
- Uses Gmail's internalDate (when Gmail received the message), not the
  sender-provided Date header.
- Retrieves message headers and attachment metadata only.
- Does NOT download attachments, classify messages, mark them read,
  apply labels, move, archive, or delete messages.

Place this file in the project root:
    E:\\ai-email-automation\\email_intake.py

Expected auth module:
    E:\\ai-email-automation\\services\\google_auth.py
"""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

from services.google_auth import get_google_services


# Strict cutoff: 24 November 2026, 13:00:00 Asia/Kolkata.
# Gmail internalDate is milliseconds since Unix epoch.
CUTOFF_TIMESTAMP_MS = 1795505400000
LOCAL_TIMEZONE = ZoneInfo("Asia/Kolkata")

# Gmail's server-side search narrows the candidate set to the cutoff date
# and later. Python then applies the exact timestamp boundary.
GMAIL_QUERY = "in:inbox after:1795505400"


def _walk_parts(payload: Dict[str, Any]):
    """Yield MIME parts recursively, including nested multipart parts."""
    if not payload:
        return

    yield payload
    for part in payload.get("parts", []) or []:
        yield from _walk_parts(part)


def _get_headers(payload: Dict[str, Any]) -> Dict[str, str]:
    headers = {}
    for header in payload.get("headers", []) or []:
        name = header.get("name", "").lower()
        if name in {"from", "to", "subject", "date"}:
            headers[name] = header.get("value", "")
    return headers


def _get_attachment_metadata(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    attachments = []

    for part in _walk_parts(payload):
        filename = part.get("filename") or ""
        body = part.get("body", {}) or {}
        attachment_id = body.get("attachmentId")

        # Include named attachments and inline named files. Do not download.
        if filename:
            attachments.append(
                {
                    "filename": filename,
                    "mime_type": part.get("mimeType", "application/octet-stream"),
                    "size_bytes": body.get("size", 0),
                    "attachment_id": attachment_id,
                }
            )

    return attachments


def _format_received_time(internal_date_ms: int) -> str:
    received_utc = datetime.fromtimestamp(
        internal_date_ms / 1000, tz=timezone.utc
    )
    return received_utc.astimezone(LOCAL_TIMEZONE).isoformat()


def fetch_eligible_inbox_messages() -> List[Dict[str, Any]]:
    """
    Fetch metadata for inbox messages strictly after the configured cutoff.

    Returns a list of message metadata dictionaries. No Gmail state is changed.
    """
    gmail, _drive = get_google_services()

    results: List[Dict[str, Any]] = []
    page_token = None
    scanned_count = 0
    skipped_at_or_before_cutoff = 0

    while True:
        request_args = {
            "userId": "me",
            "q": GMAIL_QUERY,
            "maxResults": 100,
        }
        if page_token:
            request_args["pageToken"] = page_token

        response = gmail.users().messages().list(**request_args).execute()
        message_refs = response.get("messages", []) or []

        for message_ref in message_refs:
            scanned_count += 1
            message_id = message_ref["id"]

            # Metadata format avoids retrieving the email body.
            message = (
                gmail.users()
                .messages()
                .get(
                    userId="me",
                    id=message_id,
                    format="full",
                    metadataHeaders=["From", "To", "Subject", "Date"],
                )
                .execute()
            )

            internal_date_ms = int(message.get("internalDate", "0"))

            # Strictly greater-than: exactly 13:00:00 is excluded.
            if internal_date_ms <= CUTOFF_TIMESTAMP_MS:
                skipped_at_or_before_cutoff += 1
                continue

            payload = message.get("payload", {}) or {}
            headers = _get_headers(payload)

            results.append(
                {
                    "message_id": message_id,
                    "thread_id": message.get("threadId"),
                    "internal_date_ms": internal_date_ms,
                    "received_at_ist": _format_received_time(internal_date_ms),
                    "from": headers.get("from", ""),
                    "to": headers.get("to", ""),
                    "subject": headers.get("subject", ""),
                    "sender_date_header": headers.get("date", ""),
                    "attachments": _get_attachment_metadata(payload),
                }
            )

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    # Privacy-safe diagnostics only. Do not print message IDs, sender addresses,
    # subjects, or attachment filenames.
    print("\nGmail intake completed (READ-ONLY).")
    print(f"Messages scanned: {scanned_count}")
    print(f"Skipped at/before cutoff: {skipped_at_or_before_cutoff}")
    print(f"Eligible messages: {len(results)}")
    print(
        "Cutoff: "
        + datetime.fromtimestamp(
            CUTOFF_TIMESTAMP_MS / 1000, tz=timezone.utc
        ).astimezone(LOCAL_TIMEZONE).isoformat()
    )

    return results


if __name__ == "__main__":
    fetch_eligible_inbox_messages()
