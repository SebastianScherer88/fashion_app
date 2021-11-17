# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 17:56:26 2021

@author: bettmensch
"""

import sys
sys.path.append('../ml_model_development/code')
sys.path.append('../gcp_utils')
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

from gcs_utils import read_table_from_gcs, write_table_to_gcs
from pubsub_utils import sub, pub
from google.cloud.pubsub_v1.subscriber import message

# local
MODEL_PATH = os.path.join('..','ml_model_development','artifacts','cnn_model')
image_classifier = load_model(MODEL_PATH)

META_DATA_PATH = os.path.join('..','ml_model_development','meta','target_label_mapping.csv')
labels = pd.read_csv(META_DATA_PATH)['label']

# GCP
PROJECT = 'vector-fashion-ml'
ANALYSIS_REQUEST_SUBSCRIPTION = 'read-analysis-requests'

BUCKET_NAME = 'vector-fashion-database'
BUCKET_DATA_PREFIX = 'fashion_request_data'

ANALYSIS_OUTPUT_TOPIC = 'image-analysis-outputs'

# environment variables
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../vector-fashion-ml-key.json'

def serve_analysis_request(message: message.Message) -> None:
    
    # load image file from gcs
    image_table = read_table_from_gcs(bucket_name = BUCKET_NAME,
                                      gcs_file_path = message.attributes['gcs_file_path'])
    
    # generally, we would use the same preprocessing function that was used during model build, but this is just a toy example
    image_preprocessed = np.expand_dims(image_table.values.reshape((28,28)) / 255, axis=[0,-1])
    
    # generate conditional class probabilities, i.e. apply model
    prediction = image_classifier.predict(image_preprocessed)
    
    prediction_df = pd.DataFrame(prediction[0],
                                 columns=['probability'])
    prediction_df['label'] = labels
    
    prediction_df = prediction_df.sort_values('probability',ascending=False)
                        
    prediction_df['probability'] = prediction_df['probability'].astype(str)
    predicted_class = prediction_df.iloc[0]. \
                                    to_dict()
    predicted_class['unique_id'] = message.attributes['unique_id']
                    
    # export prediction table to gcs - same location as input image
    prediction_gcs_file_path = '/'.join([message.attributes['gcs_prefix'],'model_prediction.csv'])
    
    write_table_to_gcs(table = prediction_df,
                       bucket_name = BUCKET_NAME,
                       gcs_file_path = prediction_gcs_file_path)
    
    # publish result back into designated analysis output topic on pubsub
    pub(project_id = PROJECT, 
        topic_id = ANALYSIS_OUTPUT_TOPIC,
        data = 'Model prediction ' + message.attributes['gcs_prefix'],
        **predicted_class)
    
    #print(f"Received {message}.")
    # Acknowledge the message. Unack'ed messages will be redelivered.
    message.ack()
    print(f"Acknowledged {message.attributes['unique_id']}.")
    
    return

def serve_analysis_requests():
        
    sub(project_id = PROJECT, 
        subscription_id = ANALYSIS_REQUEST_SUBSCRIPTION,
        callback = serve_analysis_request)
    
if __name__ == '__main__':
    serve_analysis_requests()
    