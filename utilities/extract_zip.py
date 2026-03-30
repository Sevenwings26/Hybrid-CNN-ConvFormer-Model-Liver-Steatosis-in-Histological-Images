import zipfile
import os

zip_path = r"HEPASS_algorithm_dataset\Original Images\TRAIN.zip"   # path to your zip file
extract_path = r"HEPASS_algorithm_dataset\Original Images\TRAIN_extracted"   # folder to extract into

# Create folder if it doesn't exist
os.makedirs(extract_path, exist_ok=True)

# Unzip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# print("Dataset extracted to:", extract_path)
