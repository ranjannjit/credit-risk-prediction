import os
import zipfile
import shutil
import subprocess

# -----------------------------
# CONFIG
# -----------------------------
DATASET = "wordsforthewise/lending-club"
ZIP_FILE = "lending-club.zip"
TARGET_FOLDER = "data"
FINAL_NAME = "lending_club.csv"

# -----------------------------
# STEP 1: Ensure data folder exists
# -----------------------------
os.makedirs(TARGET_FOLDER, exist_ok=True)

# -----------------------------
# STEP 2: Download dataset from Kaggle
# -----------------------------
print("Downloading dataset from Kaggle...")
subprocess.run([
    "kaggle", "datasets", "download",
    "-d", DATASET
], check=True)

# -----------------------------
# STEP 3: Unzip dataset
# -----------------------------
print("Extracting dataset...")
with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
    zip_ref.extractall("temp_data")

# -----------------------------
# STEP 4: Locate the correct CSV
# -----------------------------
source_file = None
for root, dirs, files in os.walk("temp_data"):
    for file in files:
        if file == "accepted_2007_to_2018Q4.csv":
            source_file = os.path.join(root, file)

if source_file is None:
    raise FileNotFoundError("accepted_2007_to_2018Q4.csv not found!")

# -----------------------------
# STEP 5: Move + Rename
# -----------------------------
destination = os.path.join(TARGET_FOLDER, FINAL_NAME)

print(f"Moving and renaming file to {destination}...")
shutil.move(source_file, destination)

# -----------------------------
# STEP 6: Cleanup
# -----------------------------
shutil.rmtree("temp_data")
os.remove(ZIP_FILE)

print("✅ Dataset ready at:", destination)