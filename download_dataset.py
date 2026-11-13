import kagglehub
import shutil
import os

# Download dataset dari Kaggle
path = kagglehub.dataset_download(
    "asutoshprad/floor-plan-dataset"
)

print("Dataset berhasil di-download!")
print("Lokasi dataset:", path)

# Salin dataset ke folder project
destination = "dataset/raw"

os.makedirs(destination, exist_ok=True)

shutil.copytree(
    path,
    destination,
    dirs_exist_ok=True
)

print("Dataset berhasil disimpan di folder dataset/raw!")