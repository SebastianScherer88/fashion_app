# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 20:16:44 2021

@author: bettmensch
"""

import os
from message_broker_api import Switchboard
import argparse

# environment variables
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join('..','gcp_pubsub','vector-fashion-ml-key.json')

def read_ml_responses(message_broker,
                      local_mode):
    
    print('Using message broker: ' + message_broker)
    print('Local mode (only applicable for kafka): ' + str(local_mode))
    
    switchboard = Switchboard(message_broker,
                              local_mode)
    
    switchboard.read_classification_responses()
    
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
    
    args = parser.parse_args()

    read_ml_responses(args.message_broker,
                      args.local_mode)