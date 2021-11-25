# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 17:28:32 2021

@author: bettmensch
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 17:56:26 2021

@author: bettmensch
"""

import sys
sys.path.append('..')

from kafka_utils import *
from kafka import KafkaConsumer

# local
LOCAL_MODE = True

if LOCAL_MODE:
    BOOTSTRAP_SERVER = 'localhost'
else:
    BOOTSTRAP_SERVER = GCP_BOOTSTRAP_SERVER

def read_analysis_responses():
    
    consumer = KafkaConsumer(KAFKA_RESPONSE_TOPIC, 
                             bootstrap_servers=[':'.join([BOOTSTRAP_SERVER,PORT])], api_version=(0, 10), 
                             consumer_timeout_ms=1000,
                             group_id='read-ml-requests',auto_offset_reset='earliest')
        
    while True:
        # for each consumed message, produce the ML model response    
        for msg in consumer:
            
            print(msg.key)
    
if __name__ == '__main__':
    read_analysis_responses()