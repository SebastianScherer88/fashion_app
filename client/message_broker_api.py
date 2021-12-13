# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 17:48:01 2021

@author: bettmensch
"""

from datetime import datetime
from string import ascii_lowercase, ascii_uppercase, digits
import random
import os

import sys
sys.path.append(os.path.join('..','kafka_system'))
sys.path.append(os.path.join('..','gcp_pubsub','gcp_utils'))

from kafka_utils import dataframe_to_json, publish_message
from gcs_utils import write_table_to_gcs
from pubsub_utils import pub

from google.cloud.pubsub_v1 import PublisherClient as Pub
from kafka import KafkaProducer

from pubsub_utils import sub
from kafka import KafkaConsumer


def get_unique_identifier():
    
    time_stamp = str(datetime.now()).replace('-','_').replace(' ','__').replace(':','_').replace('.','_')
    random_id = ''.join(random.sample(ascii_lowercase + ascii_uppercase + digits,10))
    
    full_id = '_'.join([time_stamp,random_id])
    
    return full_id

class Switchboard(object):
    '''Util class that interfaces between the 
    - client making image analysis requests and reading the analysed messages from the message broker, and 
    - the message broker itself (google pubsub or kafka)
    
    Could be extended to support reading pubsub messages with payloads so as to also use it for the ML server, but that require more work and
    is probably not as important given the use case and the fact that the ML service very customisable as its owned by ML engineers (i.e. this team).'''
    
    def __init__(self,
                 message_broker: str = 'kafka',
                 kafka_local_mode: bool = False
                 ):
        
        assert message_broker in ['pubsub','kafka']
        
        self.message_broker = message_broker
        
        # --- settings
        # GCP - general
        self.GCP_PROJECT = 'vector-fashion-ml'
        
        # pubsub settings
        # GCP - pubsub
        self.GCP_REQUEST_PUBLISH_TOPIC = 'image-analysis-requests'
        self.GCP_ANALYSIS_OUTPUT_SUBSCRIPTION = 'read-analysis-outputs'
        self.GCP_ANALYSIS_OUTPUT_GROUP = 'read-ml-requests'
        self.GCP_BUCKET_NAME = 'vector-fashion-database'
        self.GCP_BUCKET_DATA_PREFIX = 'fashion_request_data'

        # kafka settings
        # GCP - kafka
        self.KAFKA_LOCAL_MODE = kafka_local_mode

        if self.KAFKA_LOCAL_MODE :
            self.BOOTSTRAP_SERVER = 'localhost'
        else:
            self.BOOTSTRAP_SERVER = '34.147.112.135'

        # default port of main kafka node
        self.KAFKA_PORT = '9092'
        
        # topic containig image analysis requests    
        self.KAFKA_REQUEST_TOPIC = 'analysis-requests'

        # topic containing model predictions
        self.KAFKA_RESPONSE_TOPIC = 'analysis-responses'
                
        # --- publisher client instance
        if self.message_broker == 'kafka':
            self.client = KafkaProducer(bootstrap_servers=[':'.join([self.BOOTSTRAP_SERVER,self.KAFKA_PORT])], api_version=(0, 10))
            
        elif self.message_broker == 'pubsub':
            self.client = Pub()
        
    def make_classification_request(self,
                                    image):
        '''Takes a one row dataframe representing the image array and makes a single image analysis request to the appropriate message broker.'''
        
        unique_id = get_unique_identifier()
        
        if self.message_broker == 'kafka':
            random_image_value = dataframe_to_json(image)
            
            unique_id_key = bytes(unique_id, encoding='utf-8')
            
            # publish to analysis request topic
            publish_message(self.client, self.KAFKA_REQUEST_TOPIC, key=unique_id_key, value=random_image_value)
            
        elif self.message_broker == 'pubsub':
            # write to gcs storage under unique time stamp based id
            gcs_prefix = '/'.join([self.GCP_BUCKET_DATA_PREFIX,unique_id])
            gcs_file_path = '/'.join([gcs_prefix,'image_table.csv'])
            
            write_table_to_gcs(table = image,
                               bucket_name = self.GCP_BUCKET_NAME,
                               gcs_file_path = gcs_file_path)
            
            # communicate to pubsub hub that image is awaiting analysis from ML model
            pub(project_id = self.GCP_PROJECT, 
                topic_id = self.GCP_REQUEST_PUBLISH_TOPIC,
                client = self.client,
                data = 'Image request ' + unique_id,
                #test_index = str(random_sample),
                unique_id = unique_id,
                gcs_prefix = gcs_prefix,
                gcs_file_path = gcs_file_path)
            
        return unique_id
    
    def read_classification_responses(self):
        '''Reads and prints out analysed image messages from a constant stream from the appropriate message broker.'''
        
        if self.message_broker == 'kafka':
            consumer = KafkaConsumer(self.KAFKA_RESPONSE_TOPIC, 
                                     bootstrap_servers=[':'.join([self.BOOTSTRAP_SERVER,self.KAFKA_PORT])], api_version=(0, 10), 
                                     consumer_timeout_ms=1000,
                                     group_id='read-ml-requests',auto_offset_reset='earliest')
                
            while True:
                # for each consumed message, produce the ML model response    
                for msg in consumer:
                    
                    print(msg.key)                
        
        elif self.message_broker == 'pubsub':
            sub(project_id = self.GCP_PROJECT, 
                subscription_id = self.GCP_ANALYSIS_OUTPUT_SUBSCRIPTION)
            
            
        return
