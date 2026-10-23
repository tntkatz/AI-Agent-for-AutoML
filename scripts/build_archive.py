import os
import zipfile

MAX_SIZE_MB = 100.0
ARCHIVE_NAME = 'submission.zip'

def main():
    if not os.path.exists('requirements.txt'):
        print("Error: The script must be run from the project root.")
        return

    files_to_zip = []
    for item in os.listdir('.'):
        if os.path.isfile(item):
            if item.endswith('.py') or item == 'requirements.txt':
                files_to_zip.append(item)

    if 'solution.py' not in files_to_zip or 'requirements.txt' not in files_to_zip:
        print("Error: 'solution.py' and 'requirements.txt' are mandatory for the submission.")
        return

    print(f"Adding files to archive: {', '.join(files_to_zip)}")

    with zipfile.ZipFile(ARCHIVE_NAME, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in files_to_zip:
            zf.write(file, arcname=file) 

    size_bytes = os.path.getsize(ARCHIVE_NAME)
    size_mb = size_bytes / (1024 * 1024)

    print(f"Success: Archive '{ARCHIVE_NAME}' has been generated.")
    print(f"Final size: {size_mb:.4f} MB (Limit: {MAX_SIZE_MB} MB)")

    if size_mb > MAX_SIZE_MB:
        print("Critical Warning: Archive size exceeds the 100 MB platform limit.")
    else:
        print("Size validation passed.")

if __name__ == '__main__':
    main()