# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 16:03:27 2021

@author: bettmensch
"""

# Imports the Google Cloud client library
import pandas as pd
from google.cloud import storage

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