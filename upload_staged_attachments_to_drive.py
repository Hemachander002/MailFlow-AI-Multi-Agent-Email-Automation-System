"""
Upload staged Gmail attachments to app-managed Google Drive folders.

Reads the per-email staging layout created by download_eligible_attachments.py:
data/attachments/staging/<gmail-message-id>/<original-filename>

The original attachment filename is used as the Google Drive filename.
Gmail message ID + attachment index are stored as Drive appProperties so
same-named attachments from different emails are not mistaken for duplicates.
"""

import json
import mimetypes
from pathlib import Path
from typing import Dict, Optional

from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from services.google_auth import get_google_services


PROJECT_DIR = Path(__file__).resolve().parent
STAGING_DIR = PROJECT_DIR / "data" / "attachments" / "staging"
MANIFEST_PATH = PROJECT_DIR / "data" / "attachments" / "drive_upload_manifest.json"
FOLDER_CONFIG_PATH = PROJECT_DIR / "data" / "attachments" / "drive_folders.json"

RESUME_FOLDER_NAME = "Resume Attachments - AI Email Automation"
INVOICE_FOLDER_NAME = "Invoice Attachments - AI Email Automation"

PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp"
}
METADATA_FILENAME = "_attachment_metadata.json"


def load_json_file(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def save_json_file(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp_path.replace(path)


def find_app_folder(drive, folder_name: str) -> Optional[dict]:
    escaped = folder_name.replace("\\", "\\\\").replace("'", "\\'")
    query = (
        "mimeType = 'application/vnd.google-apps.folder' and "
        f"name = '{escaped}' and trashed = false"
    )
    response = drive.files().list(
        q=query,
        pageSize=100,
        fields="files(id,name,mimeType,parents)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    matches = response.get("files", [])
    return matches[0] if matches else None


def create_app_folder(drive, folder_name: str) -> dict:
    return drive.files().create(
        body={
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
        },
        fields="id,name,mimeType,parents",
        supportsAllDrives=True,
    ).execute()


def get_or_create_app_folder(drive, folder_name: str, saved_id: Optional[str]):
    if saved_id:
        try:
            folder = drive.files().get(
                fileId=saved_id,
                fields="id,name,mimeType,trashed",
                supportsAllDrives=True,
            ).execute()
            if (
                not folder.get("trashed")
                and folder.get("mimeType") == "application/vnd.google-apps.folder"
            ):
                print(f"[DRIVE] Reusing folder: {folder['name']} ({saved_id})")
                return folder
        except HttpError:
            pass

    folder = find_app_folder(drive, folder_name)
    if folder:
        print(f"[DRIVE] Found app folder: {folder['name']} ({folder['id']})")
        return folder

    folder = create_app_folder(drive, folder_name)
    print(f"[DRIVE] Created folder: {folder['name']} ({folder['id']})")
    return folder


def get_app_folders(drive) -> Dict[str, dict]:
    config = load_json_file(FOLDER_CONFIG_PATH, {})
    if not isinstance(config, dict):
        config = {}

    resume = get_or_create_app_folder(
        drive, RESUME_FOLDER_NAME, config.get("resume_folder_id")
    )
    invoice = get_or_create_app_folder(
        drive, INVOICE_FOLDER_NAME, config.get("invoice_folder_id")
    )

    save_json_file(FOLDER_CONFIG_PATH, {
        "resume_folder_id": resume["id"],
        "resume_folder_name": resume["name"],
        "invoice_folder_id": invoice["id"],
        "invoice_folder_name": invoice["name"],
    })
    return {"resume": resume, "invoice": invoice}


def get_destination(file_path: Path, folders: Dict[str, dict]):
    suffix = file_path.suffix.lower()
    if suffix in PDF_EXTENSIONS:
        return {
            "folder_id": folders["resume"]["id"],
            "folder_label": folders["resume"]["name"],
            "mime_type": "application/pdf",
        }
    if suffix in IMAGE_EXTENSIONS:
        mime_type, _ = mimetypes.guess_type(file_path.name)
        return {
            "folder_id": folders["invoice"]["id"],
            "folder_label": folders["invoice"]["name"],
            "mime_type": mime_type or "application/octet-stream",
        }
    return None


def find_existing_attachment(
    drive, message_id: str, attachment_index: int, folder_id: str
):
    """Find this exact Gmail attachment, not merely a file with the same name."""
    safe_message_id = message_id.replace("\\", "\\\\").replace("'", "\\'")
    index_text = str(attachment_index)
    query = (
        f"'{folder_id}' in parents and trashed = false and "
        "appProperties has { key='gmailMessageId' and "
        f"value='{safe_message_id}' }} and "
        "appProperties has { key='attachmentIndex' and "
        f"value='{index_text}' }}"
    )
    response = drive.files().list(
        q=query,
        pageSize=10,
        fields="files(id,name,mimeType,webViewLink,parents,appProperties)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    matches = response.get("files", [])
    return matches[0] if matches else None


def upload_one_file(
    drive,
    file_path: Path,
    destination: dict,
    message_id: str,
    attachment_index: int,
):
    existing = find_existing_attachment(
        drive, message_id, attachment_index, destination["folder_id"]
    )
    if existing:
        print(
            f"[SKIP] This email attachment is already in Drive: "
            f"{existing['name']} (Drive ID: {existing['id']})"
        )
        return existing, False

    media = MediaFileUpload(
        str(file_path),
        mimetype=destination["mime_type"],
        resumable=True,
    )
    uploaded = drive.files().create(
        body={
            "name": file_path.name,
            "parents": [destination["folder_id"]],
            "description": (
                "Uploaded by the local multi-agent email automation workflow."
            ),
            "appProperties": {
                "gmailMessageId": message_id,
                "attachment_index": str(attachment_index),
            },
        },
        media_body=media,
        fields="id,name,mimeType,webViewLink,parents,appProperties",
        supportsAllDrives=True,
    ).execute()

    print(
        f"[UPLOADED] {uploaded['name']} -> {destination['folder_label']} "
        f"(Drive ID: {uploaded['id']})"
    )
    return uploaded, True


def iter_staged_attachments():
    """Yield attachment records from message subfolders; ignore old flat files."""
    for message_dir in sorted(STAGING_DIR.iterdir()):
        if not message_dir.is_dir():
            continue

        metadata_path = message_dir / METADATA_FILENAME
        metadata = load_json_file(metadata_path, {})
        message_id = metadata.get("message_id") or message_dir.name
        attachments = metadata.get("attachments", [])

        for item in attachments:
            filename = item.get("saved_filename")
            if not filename:
                continue
            file_path = message_dir / filename
            if not file_path.is_file():
                print(f"[SKIP] Staged file is missing: {file_path.name}")
                continue

            yield {
                "file_path": file_path,
                "message_id": message_id,
                "attachment_index": int(item.get("attachment_index", 0)),
                "original_filename": item.get("original_filename", filename),
            }


def main() -> None:
    if not STAGING_DIR.is_dir():
        print(f"[ERROR] Staging folder does not exist: {STAGING_DIR}")
        print("Run download_eligible_attachments.py first.")
        return

    staged = list(iter_staged_attachments())
    if not staged:
        print("[INFO] No new-format staged attachments found.")
        print("Run the updated download_eligible_attachments.py first.")
        print("Older flat files are intentionally ignored.")
        return

    print(f"[INFO] Found {len(staged)} staged attachment(s).")
    _gmail, drive = get_google_services()

    try:
        folders = get_app_folders(drive)
    except HttpError as exc:
        status = getattr(exc.resp, "status", "unknown")
        print(f"[DRIVE ERROR] Could not prepare app folders (HTTP {status}).")
        print(f"Google API detail: {exc}")
        return

    manifest = load_json_file(MANIFEST_PATH, {"uploads": []})
    if not isinstance(manifest, dict) or not isinstance(manifest.get("uploads"), list):
        manifest = {"uploads": []}

    uploaded_count = 0
    existing_count = 0
    skipped_count = 0
    failed_count = 0

    for item in staged:
        file_path = item["file_path"]
        destination = get_destination(file_path, folders)

        if destination is None:
            skipped_count += 1
            print(f"[SKIP] Unsupported file extension: {file_path.name}")
            continue

        try:
            drive_file, was_uploaded = upload_one_file(
                drive,
                file_path,
                destination,
                item["message_id"],
                item["attachment_index"],
            )

            if was_uploaded:
                uploaded_count += 1
            else:
                existing_count += 1

            manifest["uploads"] = [
                record for record in manifest["uploads"]
                if not (
                    record.get("gmail_message_id") == item["message_id"]
                    and record.get("attachment_index") == item["attachment_index"]
                )
            ]
            manifest["uploads"].append({
                "gmail_message_id": item["message_id"],
                "attachment_index": item["attachment_index"],
                "original_filename": item["original_filename"],
                "local_filename": file_path.name,
                "drive_file_id": drive_file["id"],
                "drive_folder": destination["folder_label"],
                "drive_folder_id": destination["folder_id"],
                "mime_type": drive_file.get("mimeType", destination["mime_type"]),
                "web_view_link": drive_file.get("webViewLink"),
            })
            save_json_file(MANIFEST_PATH, manifest)

        except HttpError as exc:
            failed_count += 1
            status = getattr(exc.resp, "status", "unknown")
            print(f"[ERROR] Upload failed for {file_path.name} (HTTP {status}).")
            print(f"Google API detail: {exc}")
        except Exception as exc:
            failed_count += 1
            print(
                f"[ERROR] Upload failed for {file_path.name}: "
                f"{type(exc).__name__}: {exc}"
            )

    print("\n========== DRIVE UPLOAD SUMMARY ==========")
    print(f"New uploads: {uploaded_count}")
    print(f"Already uploaded: {existing_count}")
    print(f"Skipped unsupported files: {skipped_count}")
    print(f"Failed uploads: {failed_count}")
    print(f"Upload manifest: {MANIFEST_PATH}")
    print("Gmail messages were not modified.")
    print("===========================================\n")


if __name__ == "__main__":
    main()


