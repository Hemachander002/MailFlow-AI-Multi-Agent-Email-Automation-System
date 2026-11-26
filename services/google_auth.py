"""
Google OAuth helper for the local email automation project.

Expected project layout:
    E:\ai-email-automation\
        credentials.json
        token.json              (created automatically; keep private)
        services\
            google_auth.py

Scopes:
    Gmail: read and modify messages (labels/read state)
    Drive: create/manage files created or opened by this app
"""

from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


BASE_DIR = Path(__file__).resolve().parent.parent
CLIENT_SECRET_FILE = BASE_DIR / "credentials.json"
TOKEN_FILE = BASE_DIR / "token.json"

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/drive.file",
]


def get_google_credentials():
    """Load, refresh, or obtain Google OAuth credentials."""
    if not CLIENT_SECRET_FILE.is_file():
        raise FileNotFoundError(
            f"OAuth credentials file not found: {CLIENT_SECRET_FILE}"
        )

    credentials = None

    if TOKEN_FILE.is_file():
        print("[GOOGLE AUTH] Loading saved OAuth token...", flush=True)
        credentials = Credentials.from_authorized_user_file(
            str(TOKEN_FILE),
            SCOPES
        )

    if credentials and credentials.expired and credentials.refresh_token:
        print("[GOOGLE AUTH] Refreshing expired access token...", flush=True)
        credentials.refresh(Request())

    if not credentials or not credentials.valid:
        print("[GOOGLE AUTH] Starting browser authorization...", flush=True)
        flow = InstalledAppFlow.from_client_secrets_file(
            str(CLIENT_SECRET_FILE),
            SCOPES
        )
        credentials = flow.run_local_server(
            host="localhost",
            port=0,
            access_type="offline",
            prompt="consent"
        )

    TOKEN_FILE.write_text(credentials.to_json(), encoding="utf-8")
    print(f"[GOOGLE AUTH] Token saved locally: {TOKEN_FILE}", flush=True)
    return credentials


def get_google_services():
    """Build and return authenticated Gmail and Drive API clients."""
    credentials = get_google_credentials()

    print("[GOOGLE AUTH] Building Gmail API client...", flush=True)
    gmail = build(
        "gmail",
        "v1",
        credentials=credentials,
        cache_discovery=False
    )

    print("[GOOGLE AUTH] Building Google Drive API client...", flush=True)
    drive = build(
        "drive",
        "v3",
        credentials=credentials,
        cache_discovery=False
    )

    return gmail, drive


def test_google_connections():
    """Verify authenticated Gmail and Drive access with harmless read calls."""
    gmail, drive = get_google_services()

    print("\n========== GOOGLE CONNECTION TEST ==========", flush=True)

    # Gmail: verify access to the authorized mailbox.
    gmail_profile = gmail.users().getProfile(userId="me").execute()
    print("[GMAIL] SUCCESS: Gmail API connection established.", flush=True)
    print(f"[GMAIL] Authorized account: {gmail_profile.get('emailAddress')}", flush=True)
    print(f"[GMAIL] Total messages: {gmail_profile.get('messagesTotal')}", flush=True)

    # Drive: verify access to the authorized user's Drive.
    drive_about = drive.about().get(
        fields="user(displayName,emailAddress),storageQuota"
    ).execute()
    drive_user = drive_about.get("user", {})
    print("[DRIVE] SUCCESS: Google Drive API connection established.", flush=True)
    print(f"[DRIVE] Authorized account: {drive_user.get('emailAddress')}", flush=True)
    if drive_user.get("displayName"):
        print(f"[DRIVE] Display name: {drive_user.get('displayName')}", flush=True)

    print("=============================================\n", flush=True)
    return gmail, drive


if __name__ == "__main__":
    test_google_connections()
