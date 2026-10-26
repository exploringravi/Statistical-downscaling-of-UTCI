"""
download_drive.py
Downloads the downscaled GeoTIFFs (and anything else) from the Google Drive
folder your Earth Engine exports write to, straight to this project folder.

One-time setup (do this once, in the browser):
  1. https://console.cloud.google.com/apis/library/drive.googleapis.com
     Pick your project (ee-ravipandey336) at the top, then click "Enable".
  2. https://console.cloud.google.com/apis/credentials
     Same project. "+ Create credentials" -> "OAuth client ID".
     If asked to configure a consent screen first: User type "External",
     app name anything, your email as support/contact email, save through
     the defaults, add your own email under "Test users".
     Application type: "Desktop app". Name: anything (e.g. "UTCI downloader").
     Click "Create", then "Download JSON" on the client you just made.
  3. Save that downloaded file as client_secret.json in this same folder
     (next to this script).

After that, just run this script. The first run opens a browser tab once
to ask you to approve access to your own Drive; it remembers you after
that (drive_token.json), so later runs are silent.

pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib

Usage
  python download_drive.py
"""

from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import os
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

try:
    from config import DRIVE_FOLDER
except ImportError:
    DRIVE_FOLDER = "UTCI_downscaling"

SCOPES = ["https://www.googleapis.com/auth/drive"]  # was drive.readonly; trashing files
          # needs write access, so this widened scope requires one new consent (below)
HERE = Path(__file__).parent
CLIENT_SECRET_FILE = HERE / "client_secret.json"
TOKEN_FILE = HERE / os.environ.get("DRIVE_TOKEN", "drive_token_exploringravi.json")   # new account; drive_token.json = old account (ravipandey336)
OUT_DIR = HERE / DRIVE_FOLDER


def get_service():
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CLIENT_SECRET_FILE.exists():
                raise SystemExit(
                    f"Missing {CLIENT_SECRET_FILE}. See the setup steps in this "
                    "script's docstring (download an OAuth client JSON from "
                    "Google Cloud Console and save it with that name here)."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_FILE), SCOPES)
            creds = flow.run_local_server(port=0, prompt="select_account")   # always ask which Google account
        TOKEN_FILE.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds)


def find_folder_id(service, name):
    q = f"mimeType = 'application/vnd.google-apps.folder' and name = '{name}' and trashed = false"
    res = service.files().list(q=q, fields="files(id, name)").execute()
    matches = res.get("files", [])
    if not matches:
        raise SystemExit(f"No Drive folder named '{name}' found. Check DRIVE_FOLDER in config.py.")
    if len(matches) > 1:
        print(f"Note: {len(matches)} folders named '{name}' (Earth Engine sometimes makes duplicates), reading all of them.")
    return [m["id"] for m in matches]


def list_files(service, folder_id):
    files, page_token = [], None
    while True:
        res = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name, size, modifiedTime)",
            pageToken=page_token,
        ).execute()
        files.extend(res.get("files", []))
        page_token = res.get("nextPageToken")
        if not page_token:
            return files


def download_file(service, file_id, dest, expected_size):
    """Returns True if `dest` now holds a verified copy of the Drive file (downloaded just
    now, or already present with a matching size from an earlier run), False otherwise."""
    if dest.exists() and expected_size and dest.stat().st_size == int(expected_size):
        print(f"  {dest.name}: already downloaded, same size, skipped")
        return True
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(dest, "wb")
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        if status:
            print(f"  {dest.name}: {int(status.progress() * 100)}%", end="\r")
    fh.close()
    ok = (not expected_size) or dest.stat().st_size == int(expected_size)
    print(f"  {dest.name}: done ({dest.stat().st_size:,} bytes)" if ok
          else f"  {dest.name}: size mismatch after download, NOT trashing on Drive")
    return ok


def trash_file(service, file_id, name):
    # Moves to Drive's own trash (recoverable there for 30 days), never a hard delete.
    service.files().update(fileId=file_id, body={"trashed": True}).execute()
    print(f"  {name}: moved to Drive trash")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    service = get_service()
    files = [f for fid in find_folder_id(service, DRIVE_FOLDER) for f in list_files(service, fid)]
    if not files:
        print(f"Drive folder '{DRIVE_FOLDER}' is empty.")
        return
    print(f"{len(files)} file(s) in Drive folder '{DRIVE_FOLDER}', downloading to {OUT_DIR}\\")
    for f in files:
        ok = download_file(service, f["id"], OUT_DIR / f["name"], f.get("size"))
        if ok:
            trash_file(service, f["id"], f["name"])
    print(f"Verified and trashed on Drive: all {len(files)} file(s) now only live locally in {OUT_DIR}\\.")


if __name__ == "__main__":
    main()
