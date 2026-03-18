from azure.storage.blob import BlobServiceClient

# --- 1. CONFIGURATION ---
AZURE_CONNECTION_STRING = "connecttion string here"
CONTAINER_NAME = "raw-data"
FILE_PATH = "./dataset_temp/ocular_dataset.zip"

def main():
    print("Step 1: Connecting to Azure Blob Storage...")
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob="ocular_dataset.zip")
    
    print("Step 2: Uploading 2GB zip file to Azure.")
    with open(FILE_PATH, "rb") as data:
        # This securely streams the file from your hard drive to Azure in small chunks
        blob_client.upload_blob(data, overwrite=True)
        
    print("\nSUCCESS! Phase 1 Data Ingestion (Deliverable II.1) is complete.")
    print("The raw data has been preserved in your Azure Data Lake.")

if __name__ == "__main__":
    main()