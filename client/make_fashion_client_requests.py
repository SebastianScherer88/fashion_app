# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 17:44:43 2021

@author: bettmensch
"""

import os
import sys
sys.path.append(os.path.join('..','ml_model_development','code'))

import argparse
import pandas as pd
from build_model_utils import load_mnist
import random
import time

from message_broker_api import Switchboard

# environment variables
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join('..','gcp_pubsub','vector-fashion-ml-key.json')

MNIST_DATA_DIR = os.path.join('..','..','fashion-mnist','data','fashion')


def make_fashion_requests(message_broker,
                          local_mode,
                          pause_seconds):
    
    print('Using message broker: ' + message_broker)
    print('Local mode (only applicable for kafka): ' + str(local_mode))
    
    images_test, labels_test = load_mnist(MNIST_DATA_DIR,
                                            't10k')
    
    images_test_df = pd.DataFrame(images_test,
                                  columns = ['pixel_' + str(i) for i in range(28**2)])
    
    switchboard = Switchboard(message_broker,
                              local_mode)
    
    while True:
        random_sample = random.choice(range(images_test_df.shape[0]))
        image = images_test_df.iloc[[random_sample]]
        
        switchboard.make_classification_request(image)
        
        time.sleep(pause_seconds)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--message_broker",
                        "-m",
                        help="The type of message broker to be used. Can be 'kafka' or 'pubsub'.",
                        choices = ['kafka','pubsub'],
                        default = 'kafka',
                        type =  str)
    
    parser.add_argument("--local_mode",
                        action='store_true',
                        help="Only applicable for 'kafka' message broker. Set to true to host nodes on local machine. Set to false to use GCP cluster.")
    
    parser.add_argument("--pause_seconds",
                        "-p",
                        help="Pauses between subsequent image classification requests published",
                        default = 0.1,
                        type =  float)
    
    args = parser.parse_args()

    make_fashion_requests(args.message_broker,
                          args.local_mode,
                          args.pause_seconds)