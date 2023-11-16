# OS traversal 
import os 
import yaml

# Importing azure blob service
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient

if __name__ == '__main__': 
    # Infering current file path 
    current_path = os.path.dirname(os.path.abspath(__file__))

    # Read the configuration
    with open(os.path.join(current_path, 'configuration.yml'), 'r') as stream:
        conf = yaml.safe_load(stream)

    container_path = conf['AZURE_STORAGE_CREDENTIALS']['CONTAINERNAME']
    blob_path = conf['AZURE_STORAGE_CREDENTIALS']['BLOBNAME']
    account_url = conf['AZURE_STORAGE_CREDENTIALS']['ACCOUNTURL']
    storage_account_key = conf['AZURE_STORAGE_CREDENTIALS']['STORAGEACCOUNTKEY']
    path_for_download = conf['PATH_FOR_AZURE_FOLDER']

    # Reading the connection string 
    conn_string = conf['AZURE_STORAGE_CREDENTIALS']['CONSTRING']

    # Making a connection to azure blob 
    conn_blob = BlobServiceClient.from_connection_string(conn_string)

    # Infering the directory from which to download images
    dir_azure = conf['PATH_TO_AZURE_DATA']

    # Listing all the directories in connection blob
    azure_dirs = open("azure_dirs","w", encoding="utf-8")

    container = ContainerClient.from_connection_string(conn_str=conn_string, container_name=container_path)
    blob_service_client_instance = BlobServiceClient(account_url=account_url, credential=storage_account_key)

    azure_dirs.close()

    blob_service_client = BlobServiceClient.from_connection_string(conn_string)
    container_client = blob_service_client.get_container_client(container_path)

    img_blob_list = container.list_blobs(dir_azure)

    for blob in img_blob_list:
        if blob.name.endswith(".JPG"):
            adress=blob.name.split("/")[-2]
            local_image_name= blob.name.split("/")[-1]
            dir_azure=dir_azure.split("/")[-1]
            dir_name=f"{path_for_download}/{dir_azure}/{adress}"
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            img_blob_client = container_client.get_blob_client(blob=blob.name)
            local_path=f"{dir_name}/{local_image_name}"
            with open(local_path, "wb") as dst:
                dst.write(img_blob_client.download_blob().readall())    
                