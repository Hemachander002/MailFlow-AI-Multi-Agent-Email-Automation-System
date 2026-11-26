import os
import re
from services.google_auth import get_google_services
from pathlib import Path

def check_attachments_downloaded(message_id):
    gmail, _drive = get_google_services()

    staging_folder = (
        Path(r"E:\ai-email-automation\data\attachments\staging")
        / message_id
    )

    # Get the email details from Gmail
    message = gmail.users().messages().get(
        userId="me",
        id=message_id,
        format="full"
    ).execute()

    # Collect attachment filenames from the email
    expected_attachments = []

    def walk_parts(parts):
        for part in parts:
            filename = part.get("filename")

            if filename:
                expected_attachments.append(filename)

            # Check nested MIME parts
            if part.get("parts"):
                walk_parts(part["parts"])

    payload = message.get("payload", {})
    walk_parts(payload.get("parts", []))

    # No attachments found
    if not expected_attachments:
        return False

    # Check whether the staging folder exists
    if not staging_folder.exists() or not staging_folder.is_dir():
        return False

    # Get downloaded filenames
    downloaded_files = {
        file.name
        for file in staging_folder.iterdir()
        if file.is_file()
    }

    # Check that every expected attachment exists
    for filename in expected_attachments:
        if filename not in downloaded_files:
            return False

    return True
