#!/usr/bin/env python

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import random
from google.cloud import pubsub_v1
from google.cloud.pubsub_v1 import PublisherClient as Pub


def pub(project_id: str, 
        topic_id: str,
        client: Pub = None,
        data: str = '',
        **data_kwargs) -> None:
    """Publishes a message to a Pub/Sub topic."""
    # Initialize a Publisher client.
    if not client:
        client = Pub()
    # Create a fully qualified identifier of form `projects/{project_id}/topics/{topic_id}`
    topic_path = client.topic_path(project_id, topic_id)

    # When you publish a message, the client returns a future.
    api_future = client.publish(topic_path, data=data.encode('UTF-8'),**data_kwargs)
    message_id = api_future.result()

    print(f"Published to {topic_path}: {message_id}:{data_kwargs['unique_id']}")

# =============================================================================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
#     )
#     parser.add_argument("project_id", help="Google Cloud project ID")
#     parser.add_argument("topic_id", help="Pub/Sub topic ID")
# 
#     args = parser.parse_args()
#     
#     #image_vector_string = ','.join([str(random.choice(range(255))) for i in range(28**2)]) # too many characters: 2812 > 1024 = max
#     #image_vector_dict = dict([('pixel_'+str(i),str(random.choice(range(255)))) for i in range(28**2)]) # too many attributes: 784 > 100 = max
#     #we'll need to use a dedicated data storage/transfer service instead, like GCS
# 
#     pub(args.project_id, args.topic_id,data='test')
# =============================================================================
  
def callback(message: pubsub_v1.subscriber.message.Message) -> None:
    print(f"Received {message.attributes['unique_id']}.")
    # Acknowledge the message. Unack'ed messages will be redelivered.
    message.ack()
    print(f"Acknowledged {message.attributes['unique_id']}.")
    
def sub(project_id: str, subscription_id: str, timeout: float = None, callback = callback) -> None:
    """Receives messages from a Pub/Sub subscription."""
    # Initialize a Subscriber client
    subscriber_client = pubsub_v1.SubscriberClient()
    # Create a fully qualified identifier in the form of
    # `projects/{project_id}/subscriptions/{subscription_id}`
    subscription_path = subscriber_client.subscription_path(project_id, subscription_id)

    streaming_pull_future = subscriber_client.subscribe(
        subscription_path, callback=callback
    )
    print(f"Listening for messages on {subscription_path}..\n")

    try:
        # Calling result() on StreamingPullFuture keeps the main thread from
        # exiting while messages get processed in the callbacks.
        streaming_pull_future.result(timeout=timeout)
    except:  # noqa
        streaming_pull_future.cancel()  # Trigger the shutdown.
        streaming_pull_future.result()  # Block until the shutdown is complete.

    subscriber_client.close()


# =============================================================================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
#     )
#     parser.add_argument("project_id", help="Google Cloud project ID")
#     parser.add_argument("subscription_id", help="Pub/Sub subscription ID")
#     parser.add_argument(
#         "timeout", default=None, nargs="?", const=1, help="Pub/Sub subscription ID"
#     )
# 
#     args = parser.parse_args()
# 
#     sub(args.project_id, args.subscription_id, args.timeout)
# 
# =============================================================================
