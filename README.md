# Overview

GCP based asynchronous ML image classifier service. 

Two local processes simulate a client application that makes and receives requests via messages to GCP's pubsub service. 
It also exports the image payloads in the form of .csv tables to GCS for the ML service to pick up and process.

The ML service is also locally hosted. It 
- receives image processing requests via a pubsub subscription, 
- loads the files from GCS, 
- applies a pretrained keras classifier, 
- exports the prediction file to GCS
- publishes a message containing the classification results.

This last message is then picked up by the reception process of the client app.

## Setup

1. Clone this repository.
2. Rebuilt the conda environment by running `conda env create --file=conda_env.yaml`
3. Download the mnist fashion data set by cloning this repository: `git clone https://github.com/zalandoresearch/fashion-mnist.git`. Make sure this sits alongside this repository to maintain relative file path references across the project.

For all subsequent steps, you will need the conda environment activated.

## Building the image classifier (Question 1)

To build a keras model image classifier that achieves ~91% accuracy on the mnist-fashion test split, run the python script `./ml_model_development/code/build_model.py`.

It will export the model artifact as well as some results metrics to a newly created `./ml_model_development/artifacts` subdirectory. Together with `./ml_model_development/code/build_model_utils.py`, it should also allow you to retrain this basic model architecture on your own labelled images data set.

## Simulating the client application & ML image classification service (Question 2 & 3)

**You will need the `vector-fashion-ml-key.json` GCP credentials key in your repo top directory for any of this to work.**

To host the client process making the image classification requests, run the script `./fashion_client/make_analysis_requests.py` from a designated console.

To host the client process receiving the completed image classification requests, run the script `./fasion_client/get_analysis_outputs.py` from a designated console.

To host the ML service applying the trained ML model , run the script `./ml_server/image_classification_service.py` from one or more designated console.

Note that you can locally scale/parallelize this ML service by running the same above script from as many designated consoles in parallel as desired. 
Since the messaging system feeding these duplicate model processes is supported by pubsub and only the one subscription (`read-analysis-requests`), these processes will not interfere and simply process and hand back "their share" of incoming image classification requests in an asynchronous fashion.
You should notice the procesing speed of the individual ML processes drop with each new ML process that you launch, as the same input volume of requests is being shared amongst more and more ML servers.
A remote/cloud based version of this scaling, for example using GCP's `Cloud Run` and a dockerised Ml process script is of course also possible.
