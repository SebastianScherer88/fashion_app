# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 16:22:42 2021

@author: bettmensch
"""

# source: https://towardsdatascience.com/how-to-choose-the-best-keras-pre-trained-model-for-image-classification-b850ca4428d4

# imports
import os
os.chdir('C:/Users/bettmensch/GitReps/vector_ai/code')

import tensorflow as tf
#import tensorflow_datasets as tfds
import inspect
#from tqdm import tqdm

from tensorflow.data import Dataset as ds
import numpy as np
#from matplotlib import pyplot as plt
import pandas as pd

import sys
mnist_repo = os.path.join('..','..','fashion-mnist')
mnist_data_utils = os.path.join(mnist_repo,'utils')
sys.path.append(mnist_data_utils)

from mnist_reader import load_mnist

# =============================================================================
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers import MaxPool2D
# from tensorflow.keras.layers import Flatten
# from tensorflow.keras.layers import Dropout
# =============================================================================

def normalize_img(image, label, img_size):
    # Resize image to the desired img_size and normalize it
    image = tf.image.rgb_to_grayscale(image) # does nothing if already gray scale
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.
    
    # One hot encode the label
    label = tf.one_hot(label, depth=num_classes)
    
    return image, label

def preprocess_data(train, validation, batch_size, img_size):
    # Apply the normalize_img function on all train and validation data and create batches
    train_processed = train.map(lambda image, label: normalize_img(image, label, img_size))
    
    # If your data is already batched (eg, when using the image_dataset_from_directory function), remove .batch(batch_size)
    train_processed = train_processed.batch(batch_size).repeat()
    
    validation_processed = validation.map(lambda image, label: normalize_img(image, label, img_size))
    
    # If your data is already batched (eg, when using the image_dataset_from_directory function), remove .batch(batch_size)
    validation_processed = validation_processed.batch(batch_size)
    
    return train_processed, validation_processed

mnist_data_dir = os.path.join(mnist_repo,'data','fashion')

# -- [0] get data set
# load meta data
labels_mapping_df = pd.read_csv(os.path.join('..','meta','target_label_mapping.csv'))

# load data splits
images_train, labels_train = load_mnist(mnist_data_dir,
                                        'train')

images_test_validate, labels_test_validate = load_mnist(mnist_data_dir,
                                                        't10k')


images_test, labels_test = images_test_validate[:8000], labels_test_validate[:8000]
images_validate, labels_validate = images_test_validate[8000:], labels_test_validate[8000:]

# reshape into (observation index, width, height, n_channel) tensors
images_train_square = images_train.reshape((-1,28,28,1)) # -> 60k x 28 x 28
images_train_cube = np.concatenate([images_train_square,] * 3,axis=-1) # -> 60k x 28 x 28 x 3 (triple grey scale arrays to simulate color channels)

images_test_square = images_test.reshape((-1,28,28,1))
images_test_cube = np.concatenate([images_test_square,] * 3,axis=-1)

images_validate_square = images_validate.reshape((-1,28,28,1))
images_validate_cube = np.concatenate([images_validate_square,] * 3,axis=-1)

# convert to tensorflow native data set format
fashion_tf_train = ds.from_tensor_slices((images_train_cube,labels_train))
fashion_tf_test = ds.from_tensor_slices((images_test_cube,labels_test))
fashion_tf_validate = ds.from_tensor_slices((images_validate_cube,labels_validate))

#(train, validation), metadata = tfds.load('cats_vs_dogs', split=['train[:70%]', 'train[70%:]'], with_info=True, as_supervised=True)
batch_size = 32

# Number of training examples and labels
num_test = len(list(fashion_tf_test))
num_validation = len(list(fashion_tf_validate))
num_classes = len(set(labels_test))
num_iterations = int(num_test/batch_size)

# Print important info
print(f'Num test images: {num_test} \
        \nNum validation images: {num_validation} \
        \nNum classes: {num_classes} \
        \nNum iterations per epoch: {num_iterations}')

# apply preprocessing to data
test_processed_224, validation_processed_224 = preprocess_data(fashion_tf_test, fashion_tf_validate, batch_size, img_size=[224,224])
test_processed_331, validation_processed_331 = preprocess_data(fashion_tf_test, fashion_tf_validate, batch_size, img_size=[331,331])

pretrained_models_dict = dict([(tf_app[0],tf_app[1]) for tf_app in inspect.getmembers(tf.keras.applications,inspect.isfunction)])
pretrained_models = '\n'.join(list(pretrained_models_dict.keys()))
print(f'Pretrained models available through keras: {pretrained_models}')

model_benchmarks = {'model_name': [], 'num_model_params': [], 'validation_accuracy': []}

model_name = 'DenseNet201'
model = pretrained_models_dict[model_name]

# Special handling for "NASNetLarge" since it requires input images with size (331,331)
if 'NASNetLarge' in model_name:
    input_shape=(331,331,3)
    test_processed = test_processed_331
    validation_processed = validation_processed_331
else:
    input_shape=(224,224,3)
    test_processed = test_processed_224
    validation_processed = validation_processed_224
    
# load the pre-trained model with global average pooling as the last layer and freeze the model weights
pre_trained_model = model(include_top=False, pooling='avg', input_shape=input_shape)
pre_trained_model.trainable = False

# custom modifications on top of pre-trained model and fit
clf_model = tf.keras.models.Sequential()
clf_model.add(pre_trained_model)
clf_model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
clf_model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
history = clf_model.fit(test_processed,
                        epochs=1, 
                        validation_data=validation_processed, 
                        steps_per_epoch=num_iterations)

# Calculate all relevant metrics
model_benchmarks['model_name'].append(model_name)
model_benchmarks['num_model_params'].append(pre_trained_model.count_params())
model_benchmarks['validation_accuracy'].append(history.history['val_accuracy'][-1])