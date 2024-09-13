from __future__ import print_function
import os.path
import pickle
import os
import sys
import datetime
from tqdm import tqdm
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# If modifying these SCOPES, delete the file token.pickle.
SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def authenticate():
    """Authenticate the user and return the Drive service."""
    creds = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "./credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)
    return build("drive", "v3", credentials=creds)


def create_folder(service, folder_name, parent_folder_id=None):
    """Create a folder on Google Drive."""
    file_metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    if parent_folder_id:
        file_metadata["parents"] = [parent_folder_id]
    folder = service.files().create(body=file_metadata, fields="id, name").execute()
    print(f'Created folder "{folder_name}" with ID: {folder.get("id")}')
    return folder.get("id")


def upload_file(service, file_path, folder_id=None):
    """Upload a file to Google Drive."""
    file_name = os.path.basename(file_path)
    file_metadata = {"name": file_name}
    if folder_id:
        file_metadata["parents"] = [folder_id]

    media = MediaFileUpload(file_path, resumable=True)
    file = (
        service.files()
        .create(body=file_metadata, media_body=media, fields="id, name")
        .execute()
    )
    return file.get("id"), file.get("name")


def count_files_in_directory(directory_path):
    """Count total number of files in a directory and its subdirectories."""
    total_files = 0
    for root, dirs, files in os.walk(directory_path):
        total_files += len(files)
    return total_files


def upload_folder(service, folder_path, parent_drive_folder_id=None):
    """Recursively upload a folder and its contents to Google Drive."""
    folder_name = os.path.basename(folder_path)
    drive_folder_id = create_folder(service, folder_name, parent_drive_folder_id)

    total_files = count_files_in_directory(folder_path)
    progress_bar = tqdm(total=total_files, desc=f"Uploading '{folder_name}'")

    for root, dirs, files in os.walk(folder_path):
        # Upload files
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if os.path.isfile(file_path):
                file_id, file_name_on_drive = upload_file(
                    service, file_path, drive_folder_id
                )
                progress_bar.set_postfix(file=file_name_on_drive, refresh=False)
                progress_bar.update(1)
                print(f'Uploaded file "{file_path}" to Google Drive with ID: {file_id}')

        # Recurse into subfolders
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)
            upload_folder(service, subfolder_path, drive_folder_id)

    progress_bar.close()


if __name__ == "__main__":
    drive_service = authenticate()

    # Get folder path from command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python3 upload_data.py <folder_path>")
        sys.exit(1)
    folder_path = sys.argv[1].rstrip("/")  # 여기서 마지막 /를 제거

    # Validate folder path
    if not os.path.isdir(folder_path):
        print(
            f'Error: The folder path "{folder_path}" does not exist or is not a directory.'
        )
        sys.exit(1)

    # Create a unique folder name using current date
    current_date_str = datetime.datetime.now().strftime("%y%m%d")
    folder_name = f"{os.path.basename(folder_path)}_{current_date_str}"

    parent_drive_folder_id = (
        "18PqFNJLxQgP2fzMA5NcfciVJ4tEst8wC"  # Parent Google Drive folder ID
    )
    upload_folder(drive_service, folder_path, parent_drive_folder_id)

    print("\n\n***[COMPLETED !]***")
