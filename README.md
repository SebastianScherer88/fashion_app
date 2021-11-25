# Setup

1. Clone this repository.
2. Rebuilt the conda environment by running `conda env create --file=conda_env.yaml`
3. Download the mnist fashion data set by cloning this repository: `git clone https://github.com/zalandoresearch/fashion-mnist.git`. Make sure this sits alongside this repository to maintain relative file path references across the project.

For all subsequent steps, you will need the conda environment activated.

Note: If you want to run the Kafka system *locally*, you will also need to be able to run both Kafka and Zookeeper on your machine. You can run in *non-local* mode, though, so the conda environment will be enough for that case.

# Building the image classifier (Question 1)

To build a keras model image classifier that achieves ~91% accuracy on the mnist-fashion test split, run the python script `./ml_model_development/code/build_model.py`.

It will export the model artifact as well as some results metrics to a newly created `./ml_model_development/artifacts` subdirectory. Together with `./ml_model_development/code/build_model_utils.py`, it should also allow you to retrain this basic model architecture on your own labelled images data set.

# Asyncronous Micro-services application - **GCP** system (Question 2 & 3)

GCP based asynchronous ML image classifier service. 

Two local processes simulate a client application that makes and receives requests via messages to the GCP's pubsub service. 
It also exports the image payloads in the form of .csv tables to GCS for the ML service to pick up and process.

The ML service is also locally hosted. It 
- receives image processing requests via a pubsub subscription, 
- loads the files from GCS, 
- applies a pretrained keras classifier, 
- exports the prediction file to GCS
- publishes a message containing the classification results.

This last message is then picked up by the reception process of the client app.

## Steps

**You will need the `vector-fashion-ml-key.json` GCP credentials key in your repo top directory for any of this to work.**

To host the client process making the image classification requests, run `python make_analysis_requests.py` from a designated console while in the `./gcp_pubsub/fashion_client` directory.

To host the client process receiving the completed image classification requests, run `python get_analysis_outputs.py` from a designated console while in the `./gcp_pubsub/fasion_client` directory.

To host the ML service applying the trained ML model , run `python image_classification_service.py` from one or more designated console(s) while in the `./gcp_pubsub/ml_server` directory.

Note that you can locally scale/parallelize this ML service by running the same above script from as many designated consoles in parallel as desired. 
Since the messaging system feeding these duplicate model processes is supported by pubsub and only the one subscription (`read-analysis-requests`), these processes will not interfere and simply process and hand back "their share" of incoming image classification requests in an asynchronous fashion.
You should notice the output volume of the individual ML processes drop with each new ML process that you launch, as the same volume of incoming requests is being shared amongst more and more ML servers.
A remote/cloud based version of this scaling, for example using GCP's `Cloud Run` and a dockerised ML process script is of course also possible and preferable for a commercial use case.

## Sources

I had to learn a new GCP service - pubsub - and used `Priyanka Vergadia`'s great [series of videos](https://www.youtube.com/watch?v=cvu53CnZmGI).
The code supporting publisher and subscription clients was taken and modified as needed from [this repo](https://github.com/googleapis/python-pubsub/tree/main/samples/snippets/quickstart)

# Asyncronous Micro-services application - **Kafka** system (Question 2 & 3)

The same micro services application is implemented with a Kafka message broker hub instead of GCP pubsub. The upside of this is that we don't have to use an additional payload storage layer like GCS but instead can encode images into the messages directly. The downside is that we have to look after the infra and processes supporting the distributed message broker ourselves.

## Setup

It is recommended to set `LOCAL_MODE = False` in the `kafka_utils.py`. This means that the remotely hosted Kafka cluster on GCP will be used, with only the consumer/production processes running on your machine. If you want to run Kafka locally instead, set to `True` and make sure your local broker is available on port 9092.

## Steps

To create the necessary topics on your Kafka brokers (remote or local), run `create_kafka_topics.py` from the `./kafka_system` directory.

To host the client process making the image classification requests, run `python make_analysis_requests.py` from a designated console while in the `./kafka_system/fashion_client` directory.

To host the client process receiving the completed image classification requests, run `python get_analysis_outputs.py` from a designated console while in the `./kafka_system/fasion_client` directory.

To host the ML service applying the trained ML model , run `python image_classification_service.py` from one or more designated console(s) while in the `./kafka_system/ml_server` directory.

Note that you can locally scale/parallelize this ML service by running the same above script from as many designated consoles in parallel as desired. Since the group id is specified, subsequent consumer processes will be assigned remaining partitions in the topic to read from in parallel. Note that by default, only 3 partitions are created, so either increase that number at the topic creation stage or limit the ML process to 3 to avoid idle consumers.

You should notice the output volume of the individual ML processes drop with each new ML process that you launch, as the same volume of incoming requests is being shared amongst more and more ML servers.

A remote/cloud based version of this scaling, for example using GCP's `Cloud Run` and a dockerised ML process script is of course also possible and preferable for a commercial use case.

## Sources

Youtube, the [kafka-python documentation](https://kafka-python.readthedocs.io/en/master/) and stackeroverflow. Also [this instructive repo](https://github.com/kadnan/Calories-Alert-Kafka.git). GCP Kafka cluster deployed using this GCP marketplace template: (![image](https://user-images.githubusercontent.com/26059962/143345163-d89f992d-44b8-415e-83de-651d48eb7d64.png)
) :)
