# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 17:56:26 2021

@author: bettmensch
"""

import os
import sys
sys.path.append(os.path.join('..','ml_model_development','code'))

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

from kafka_utils import *
from kafka import KafkaConsumer, KafkaProducer

# local
MODEL_PATH = os.path.join('..','ml_model_development','artifacts','cnn_model')
image_classifier = load_model(MODEL_PATH)

META_DATA_PATH = os.path.join('..','ml_model_development','meta','target_label_mapping.csv')
labels = pd.read_csv(META_DATA_PATH)['label']

LOCAL_MODE = False

if LOCAL_MODE:
    BOOTSTRAP_SERVER = 'localhost'
else:
    BOOTSTRAP_SERVER = GCP_BOOTSTRAP_SERVER

def serve_analysis_requests():
    
    consumer = KafkaConsumer(KAFKA_REQUEST_TOPIC, 
                             bootstrap_servers=[':'.join([BOOTSTRAP_SERVER,PORT])], api_version=(0, 10), 
                             consumer_timeout_ms=1000,
                             group_id='read-ml-requests',auto_offset_reset='earliest')
    
    producer = KafkaProducer(bootstrap_servers=[':'.join([BOOTSTRAP_SERVER,PORT])], api_version=(0, 10))
    
    while True:
        # for each consumed message, produce the ML model response    
        for msg in consumer:
            
            print(msg.key)
            # get image tensor
            image_array = json_to_dataframe(msg.value).values.reshape((28,28)) / 255
            image_tensor = np.expand_dims(image_array, axis=[0,-1])
            
            # generate conditional class probabilities, i.e. apply model
            prediction = image_classifier.predict(image_tensor)
            
            prediction_df = pd.DataFrame(prediction[0],
                                         columns=['probability'])
            prediction_df['label'] = labels
            prediction_df['probability'] = prediction_df['probability'].astype(str)
            
            # publish image classification results
            publish_message(producer, KAFKA_RESPONSE_TOPIC, key=msg.key, value=dataframe_to_json(prediction_df))
            
if __name__ == '__main__':
    
    serve_analysis_requests()
    