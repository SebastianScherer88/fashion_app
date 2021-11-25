# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 21:10:08 2021

@author: bettmensch
"""

import os
import sys
sys.path.append(os.path.join('..','..','ml_model_development','code'))
sys.path.append('..')

import argparse
import pandas as pd
from build_model_utils import load_mnist
from kafka_utils import *
import random
from datetime import datetime
from string import ascii_lowercase, ascii_uppercase, digits
import time
from kafka import KafkaProducer

# constants
# local
MNIST_DATA_DIR = os.path.join('..','..','..','fashion-mnist','data','fashion')

LOCAL_MODE = True

if LOCAL_MODE:
    BOOTSTRAP_SERVER = 'localhost'
else:
    BOOTSTRAP_SERVER = GCP_BOOTSTRAP_SERVER

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
    
    client = KafkaProducer(bootstrap_servers=[':'.join([BOOTSTRAP_SERVER,PORT])], api_version=(0, 10))
    
    # ping randomly sample requests until process is killed
    while True:
        
        unique_id_key = bytes(get_unique_identifier(), encoding='utf-8')
        
        random_sample = random.choice(range(images_test_df.shape[0]))
        random_image_value = dataframe_to_json(images_test_df.iloc[[random_sample]])
        
        # publish to analysis request topic
        publish_message(client, KAFKA_REQUEST_TOPIC, key=unique_id_key, value=random_image_value)
        
        time.sleep(pause_seconds)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("pause_seconds", 
                        help="Pauses between subsequent image classification requests published",
                        nargs='?',
                        default = 5,
                        type =  float)

    args = parser.parse_args()

    make_fashion_requests(args.pause_seconds)