# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 17:28:32 2021

@author: bettmensch
"""

import os
import sys
sys.path.append(os.path.join('..','..','ml_model_development','code'))
sys.path.append(os.path.join('..','gcp_utils'))

from pubsub_utils import sub

# constants

# GCP
PROJECT = 'vector-fashion-ml'
ANALYSIS_OUTPUT_SUBSCRIPTION = 'read-analysis-outputs'
BUCKET_NAME = 'vector-fashion-database'
BUCKET_DATA_PREFIX = 'fashion_request_data'

# environment variables
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join('..','vector-fashion-ml-key.json')

def get_analysis_outputs():
    sub(project_id = PROJECT, 
        subscription_id = ANALYSIS_OUTPUT_SUBSCRIPTION)
    
if __name__ == '__main__':
    get_analysis_outputs()
    