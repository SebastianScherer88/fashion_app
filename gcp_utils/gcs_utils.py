# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 16:03:27 2021

@author: bettmensch
"""

# Imports the Google Cloud client library
import pandas as pd
from google.cloud import storage

# =============================================================================
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './vector-fashion-ml-key.json'
# print(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
# =============================================================================

# =============================================================================
# # Instantiates a client
# storage_client = storage.Client()
# 
# # The name for the new bucket
# bucket_name = "vector-fashion-database"
# bucket = storage_client.bucket(bucket_name)
# 
# source_file_name = 'test_file_2.txt'
# destination_blob_name = '/'.join(['test_files',source_file_name]) # "/" required to create "subdirectories" on GCS
# 
# blob = bucket.blob(destination_blob_name)
# blob.upload_from_filename(source_file_name)
# 
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'/your_GCP_creds/credentials.json'
# 
# df = pd.DataFrame(data=[{1,2,3},{4,5,6}],columns=['a','b','c'])
# 
# client = storage.Client()
# bucket = client.get_bucket('my-bucket-name')
#     
# bucket.blob('upload_test/test.csv').upload_from_string(df.to_csv(index=keep_index), 'text/csv')
# =============================================================================

def write_table_to_gcs(table: pd.DataFrame,
                       bucket_name: str,
                       gcs_file_path: str,
                       keep_index: bool = False) -> None:
    
    # create gcs service client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    table_buffer = table.to_csv(index=keep_index)
    
    bucket. \
        blob(gcs_file_path). \
        upload_from_string(table_buffer,'text/csv')
        
    return
    
def read_table_from_gcs(bucket_name: str,
                        gcs_file_path: str) -> pd.DataFrame:
    
    full_gcs_file_path = '/'.join(['gs:/',bucket_name,gcs_file_path])
    
    table = pd.read_csv(full_gcs_file_path)
    
    return table