from pathlib import Path
from io import BytesIO

from google import genai
from googleapiclient.http import MediaIoBaseDownload

from email_classifier import API_KEY, classify_gmail_message
from services.google_auth import get_google_services
from email_intake import fetch_eligible_inbox_messages
from orchestration.database import claim_email, update_email_status

from download_eligible_attachments import (
    walk_mime_parts,
    safe_filename,
    download_attachment_bytes,
    ensure_correct_extension,
)
from upload_staged_attachments_to_drive import (
    get_app_folders,
    get_destination,
    upload_one_file,
)
from agents.resume_agent_with_embeddings_json import process_resume
from scripts.invoice_pipeline import process_invoice
from extra import check_attachments_downloaded


BASE_DIR = Path(__file__).resolve().parent.parent
STAGING_DIR = BASE_DIR / "data" / "attachments" / "staging"
JOB_ID = "DS-001"


def gmail_intake_node(state):
    """Claim exactly one eligible Inbox message for this graph execution."""
    emails = fetch_eligible_inbox_messages()
    if not emails:
        return {
            "processing_status": "no_eligible_emails",
            "current_step": "gmail_intake_complete",
        }

    for item in emails:
        message_id = item["message_id"]
        if not claim_email(
            message_id=message_id,
            thread_id=item.get("thread_id"),
            received_at=item.get("received_at_ist"),
        ) and check_attachments_downloaded(message_id) == True :
            continue

        return {
            "message_id": message_id,
            "thread_id": item.get("thread_id", ""),
            "internal_date_ms": item.get("internal_date_ms"),
            "received_at_ist": item.get("received_at_ist", ""),
            "subject": item.get("subject", ""),
            "from": item.get("from", ""),
            "attachment_metadata": item.get("attachments", []),
            "current_step": "gmail_intake_complete",
            "processing_status": "email_fetched",
        }

    return {
        "processing_status": "no_unprocessed_emails",
        "current_step": "gmail_intake_complete",
    }


def classify_email_node(state):
    """Classify one Gmail message and persist its classification."""
    if not API_KEY:
        raise RuntimeError("Gemini API key is missing; check the local .env file.")

    message_id = state["message_id"]
    update_email_status(
        message_id=message_id,
        processing_status="PROCESSING",
        current_step="classification_in_progress",
    )

    try:
        gmail, _drive = get_google_services()
        client = genai.Client(api_key=API_KEY)
        result = classify_gmail_message(
            gmail=gmail,
            client=client,
            email_item=state,
        )
        category = str(result["category"]).upper()

        update_email_status(
            message_id=message_id,
            processing_status="CLASSIFIED",
            current_step="classification_complete",
            category=category,
            confidence=result["confidence"],
        )
        return {
            "category": category,
            "confidence": result["confidence"],
            "classification_reason": result.get("reason", ""),
            "needs_review": result.get("needs_review", False),
            "current_step": "classification_complete",
            "processing_status": "classified",
        }
    except Exception:
        update_email_status(
            message_id=message_id,
            processing_status="FAILED",
            current_step="classification_failed",
            error_message="Classification failed. Inspect local application logs.",
        )
        raise


def finalize_routing_node(state):
    """Persist routing for non-agent categories."""
    category = str(state.get("category", "")).upper()
    if category == "INVOICE":
        status, step = "READY_FOR_INVOICE", "invoice_route_selected"
    elif category == "APPLICATION":
        status, step = "READY_FOR_RESUME", "resume_route_selected"
    elif category in {"SPAM", "OTHER"}:
        status, step = "SKIPPED", "non_target_email_skipped"
    else:
        status, step = "REVIEW", "manual_review_required"

    update_email_status(
        message_id=state["message_id"],
        processing_status=status,
        current_step=step,
        category=category,
    )
    return {"processing_status": status, "current_step": step}


def _select_and_stage_attachment(gmail, message_id, category):
    """Download the first supported attachment matching the classified route."""
    full_message = (
        gmail.users().messages()
        .get(userId="me", id=message_id, format="full")
        .execute()
    )
    payload = full_message.get("payload", {}) or {}
    message_dir = STAGING_DIR / message_id
    message_dir.mkdir(parents=True, exist_ok=True)

    supported_index = 0
    for part in walk_mime_parts(payload):
        filename = part.get("filename") or ""
        if not filename:
            continue

        mime_type = (part.get("mimeType") or "").lower()
        is_pdf = mime_type in {"application/pdf", "application/x-pdf"}
        is_image = mime_type.startswith("image/")

        # Keep attachment indexes consistent with the existing bulk downloader:
        # indexes count every supported PDF/image attachment in MIME order.
        if not (is_pdf or is_image):
            continue
        supported_index += 1

        if category == "APPLICATION" and not is_pdf:
            continue
        if category == "INVOICE" and not is_image:
            continue

        content = download_attachment_bytes(gmail, message_id, part)
        safe_name = safe_filename(filename)
        final_name = ensure_correct_extension(safe_name, content, mime_type)
        file_path = (message_dir / final_name).resolve()
        file_path.write_bytes(content)

        return file_path, supported_index

    expected = "PDF" if category == "APPLICATION" else "image"
    raise FileNotFoundError(f"No supported {expected} attachment found for this email.")

def _select_and_stage_attachment_two(gmail, message_id, category):
    """Download the next supported attachment that is missing from staging."""

    full_message = (
        gmail.users().messages()
        .get(userId="me", id=message_id, format="full")
        .execute()
    )

    payload = full_message.get("payload", {}) or {}

    message_dir = STAGING_DIR / message_id
    message_dir.mkdir(parents=True, exist_ok=True)

    supported_index = 0

    for part in walk_mime_parts(payload):
        filename = part.get("filename") or ""
        if not filename:
            continue

        mime_type = (part.get("mimeType") or "").lower()

        is_pdf = mime_type in {"application/pdf", "application/x-pdf"}
        is_image = mime_type.startswith("image/")

        if not (is_pdf or is_image):
            continue

        supported_index += 1

        # Filter attachments based on category
        if category == "APPLICATION" and not is_pdf:
            continue

        if category == "INVOICE" and not is_image:
            continue

        # Convert Gmail filename to the same filename used by downloader
        safe_name = safe_filename(filename)

        # Download only if this attachment is not already staged
        existing_files = {
            file.name.lower()
            for file in message_dir.iterdir()
            if file.is_file()
        }

        # Check by sanitized filename first
        if safe_name.lower() in existing_files:
            continue

        # Download attachment
        content = download_attachment_bytes(gmail, message_id, part)

        final_name = ensure_correct_extension(
            safe_name,
            content,
            mime_type
        )

        # Check again using the final filename
        file_path = message_dir / final_name

        if file_path.exists():
            continue

        file_path.write_bytes(content)

        return file_path.resolve(), supported_index

    expected = "PDF" if category == "APPLICATION" else "image"

    raise FileNotFoundError(
        f"No remaining {expected} attachments to download for this email."
    )


def _download_drive_file(drive, file_id, destination_path):
    request = drive.files().get_media(fileId=file_id)
    with destination_path.open("wb") as handle:
        downloader = MediaIoBaseDownload(handle, request)
        done = False
        while not done:
            _status, done = downloader.next_chunk()
    return destination_path


def process_specialist_agent_node(state):
    """Upload the selected attachment to Drive, download it back, then invoke its agent."""
    message_id = state["message_id"]
    category = str(state.get("category", "")).upper()

    try:
        gmail, drive = get_google_services()
        staged_path, attachment_index = _select_and_stage_attachment_two(
            gmail, message_id, category
        )

        folders = get_app_folders(drive)
        destination = get_destination(staged_path, folders)
        if not destination:
            raise ValueError("Attachment type has no configured Drive destination.")

        drive_file, _uploaded_now = upload_one_file(
            drive=drive,
            file_path=staged_path,
            destination=destination,
            message_id=message_id,
            attachment_index=attachment_index,
        )

        # The specialist receives a local copy downloaded from Drive, not the
        # original Gmail staging path.
        agent_dir = staged_path.parent / "from_drive"
        agent_dir.mkdir(parents=True, exist_ok=True)
        drive_path = agent_dir / staged_path.name
        _download_drive_file(drive, drive_file["id"], drive_path)

        update_email_status(
            message_id=message_id,
            processing_status="PROCESSING",
            current_step="specialist_agent_in_progress",
            category=category,
        )

        if category == "INVOICE":
            agent_result = process_invoice(drive_path)
        elif category == "APPLICATION":
            agent_result = process_resume(drive_path, JOB_ID)
        else:
            raise ValueError(f"Unsupported specialist route: {category}")

        # Keep the graph/checkpoint result free of email body or candidate data.
        safe_result = {}
        if isinstance(agent_result, dict):
            for key in ("status", "invoice_id", "success", "job_id"):
                if key in agent_result:
                    safe_result[key] = str(agent_result[key])
        return {
            "drive_file_ids": [drive_file["id"]],
            "drive_file_links": [drive_file.get("webViewLink", "")],
            "local_attachment_paths": [str(drive_path)],
            "agent_result": safe_result,
            "processing_status": "COMPLETED",
            "current_step": "specialist_agent_completed",
        }

    except Exception:
        update_email_status(
            message_id=message_id,
            processing_status="FAILED",
            current_step="specialist_agent_failed",
            error_message="Attachment or specialist processing failed. Inspect local logs.",
        )
        raise


def complete_agent_node(state):
    """Mark a successfully executed specialist route complete in the ledger."""
    update_email_status(
        message_id=state["message_id"],
        processing_status="COMPLETED",
        current_step="specialist_agent_completed",
        category=state.get("category"),
    )
    return {
        "processing_status": "COMPLETED",
        "current_step": "specialist_agent_completed",
    }
