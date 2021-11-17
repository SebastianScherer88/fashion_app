# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 16:44:37 2021

@author: bettmensch
"""

import sys
sys.path.append('../ml_model_development/code')
sys.path.append('../gcp_utils')
import argparse
import os
import pandas as pd
from build_model_utils import load_mnist
import random
from datetime import datetime
from string import ascii_lowercase, ascii_uppercase, digits
import time

from gcs_utils import write_table_to_gcs
from pubsub_utils import pub

# constants
# local
MNIST_DATA_DIR = os.path.join('..','..','fashion-mnist','data','fashion')

# GCP
PROJECT = 'vector-fashion-ml'
REQUEST_PUBLISH_TOPIC = 'image-analysis-requests'
BUCKET_NAME = 'vector-fashion-database'
BUCKET_DATA_PREFIX = 'fashion_request_data'

# environment variables
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../vector-fashion-ml-key.json'

def get_unique_identifier():
    
    time_stamp = str(datetime.now()).replace('-','_').replace(' ','__').replace(':','_').replace('.','_')
    random_id = ''.join(random.sample(ascii_lowercase + ascii_uppercase + digits,10))
    
    full_id = '_'.join([time_stamp,random_id])
    
    return full_id

def make_fashion_requests(pause_seconds: int = 0.1):
    
    images_test, labels_test = load_mnist(MNIST_DATA_DIR,
                                            't10k')
    
    images_test_df = pd.DataFrame(images_test,
                                  columns = ['pixel_' + str(i) for i in range(28**2)])
    
    # ping randomly sample requests until process is killed
    while True:
        random_sample = random.choice(range(images_test_df.shape[0]))
        random_image = images_test_df.iloc[[random_sample]]
        
        # write to gcs storage under unique time stamp based id
        unique_id = get_unique_identifier()
        gcs_prefix = '/'.join([BUCKET_DATA_PREFIX,unique_id])
        gcs_file_path = '/'.join([gcs_prefix,'image_table.csv'])
        
        write_table_to_gcs(table = random_image,
                           bucket_name = BUCKET_NAME,
                           gcs_file_path = gcs_file_path)
        
        # communicate to pubsub hub that image is awaiting analysis from ML model
        pub(project_id = PROJECT, 
            topic_id = REQUEST_PUBLISH_TOPIC,
            data = 'Image request ' + unique_id,
            test_index = str(random_sample),
            unique_id = unique_id,
            gcs_prefix = gcs_prefix,
            gcs_file_path = gcs_file_path)
        
        time.sleep(pause_seconds)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("pause_seconds", 
                        help="Pauses between subsequent image classification requests published",
                        type =  float)

    args = parser.parse_args()

    make_fashion_requests(args.pause_seconds)