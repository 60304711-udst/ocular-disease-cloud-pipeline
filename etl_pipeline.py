import os
import zipfile
import cv2
import numpy as np
from azure.storage.blob import BlobServiceClient
from tqdm import tqdm

# --- CONFIGURATION ---
AZURE_CONNECTION_STRING = "connetction string here"
PROCESSED_CONTAINER = "processed-data"
LOCAL_ZIP_PATH = "./dataset_temp/ocular_dataset.zip"
EXTRACT_PATH = "./dataset_temp/extracted"

def apply_clahe(image_path):
    # 1. Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
        
    # 2. Resize to a standard tensor size (224x224) for future ResNet50 models
    img_resized = cv2.resize(img, (224, 224))
    
    # 3. Apply CLAHE Feature Extraction
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl_img = clahe.apply(img_resized)
    return cl_img

def main():
    print("Step 1: Extracting ALL images from the local zip...")
    os.makedirs(EXTRACT_PATH, exist_ok=True)
    
    with zipfile.ZipFile(LOCAL_ZIP_PATH, 'r') as zip_ref:
        # Find all the image files
        image_files = [f for f in zip_ref.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        zip_ref.extractall(EXTRACT_PATH, members=image_files)

    print("\nStep 2: Connecting to Azure 'processed-data' container...")
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(PROCESSED_CONTAINER)

    print(f"\nStep 3: Applying CLAHE extraction and uploading {len(image_files)} features to Azure...")
    # tqdm gives us a nice progress bar in the terminal!
    for file_name in tqdm(image_files):
        local_file_path = os.path.join(EXTRACT_PATH, file_name)
        
        # Apply our ETL and Feature Extraction
        processed_img = apply_clahe(local_file_path)
        if processed_img is None:
            continue
            
        # Save temporarily
        temp_save_path = local_file_path.replace(".jpg", "_clahe.jpg")
        cv2.imwrite(temp_save_path, processed_img)
        
        # Upload directly to the processed zone
        blob_name = os.path.basename(temp_save_path)
        blob_client = container_client.get_blob_client(blob=blob_name)
        
        with open(temp_save_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

    print("\nSUCCESS! Full Dataset ETL and Feature Extraction is complete.")

if __name__ == "__main__":
    main()