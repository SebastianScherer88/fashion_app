# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 21:57:37 2021

@author: bettmensch
"""

from kafka.admin import KafkaAdminClient, NewTopic
from kafka_utils import *

def main():
    
    admin_client = KafkaAdminClient(
        bootstrap_servers=[':'.join([BOOTSTRAP_SERVER,PORT])], 
        client_id='topic_creator'
    )
    
    topic_list = [NewTopic(name=KAFKA_REQUEST_TOPIC, num_partitions=N_REQUEST_PARTITIONS, replication_factor=REQUEST_REP_FACTOR),
                  NewTopic(name=KAFKA_RESPONSE_TOPIC, num_partitions=N_RESPONSE_PARTITIONS, replication_factor=RESPONSE_REP_FACTOR)]
    
    admin_client.create_topics(new_topics=topic_list, validate_only=False)
    
if __name__ == '__main__':
    main()