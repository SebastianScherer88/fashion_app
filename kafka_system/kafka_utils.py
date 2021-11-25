# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 21:29:19 2021

@author: bettmensch
"""

import json
import pandas as pd

# GCP
GCP_BOOTSTRAP_SERVER = '34.147.112.135'

PORT = '9092'

LOCAL_MODE = False

if LOCAL_MODE:
    BOOTSTRAP_SERVER = 'localhost'
else:
    BOOTSTRAP_SERVER = GCP_BOOTSTRAP_SERVER

# topic containig image analysis requests    
KAFKA_REQUEST_TOPIC = 'analysis-requests'
N_REQUEST_PARTITIONS = 3 # no sense for this to be greater than number of local/GCP brokers
REQUEST_REP_FACTOR = 3 # strictly capped by number of local/GCP brokers

# topic containing model predictions
KAFKA_RESPONSE_TOPIC = 'analysis-responses'
N_RESPONSE_PARTITIONS = 3 # no sense for this to be greater than number of local/GCP brokers
RESPONSE_REP_FACTOR = 3 # capped by number of local/GCP brokers

def dataframe_to_json(dataframe):
    
    return json.dumps(dataframe.to_dict()).encode('utf-8')

def json_to_dataframe(encoded_json):
    
    dictionary = json.loads(encoded_json)
    dataframe = pd.DataFrame(dictionary)
    #dataframe.index = dataframe.index.astpye(int)
    
    return dataframe

def publish_message(producer_instance, topic_name, key, value,verbose=False):
    try:
        producer_instance.send(topic_name, key=key, value=value)
        producer_instance.flush()
        if verbose:
            print('Message ' + str(value) + ' published successfully to topic ' + str(topic_name))
        else:
            print('Message ' + str(key) + ' published successfully to topic ' + str(topic_name))
    except Exception as ex:
        print('Exception in publishing message')
        print(str(ex))