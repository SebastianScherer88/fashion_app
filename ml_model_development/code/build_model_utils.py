# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 19:26:44 2021

@author: bettmensch
"""

import os
import gzip
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset as ds

def load_mnist(path, kind='train'):
    

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def normalize_img(image, label, img_size, n_classes, convert_to_grayscale=False):
    # Resize image to the desired img_size and normalize it
    if convert_to_grayscale:
        image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.
    
    # One hot encode the label
    #label = tf.one_hot(label, depth=n_classes)
    
    return image, label

def preprocess_data(images: np.array, #[n_observations, width, height, n_channels],
                    labels: np.array, #[n_observations,], with integer encoded class indices 0,1,..., n_classes - 1
                    batch_size: int, 
                    img_size=[28,28],
                    n_classes=10):
    '''Takes images and labels in array form and converts into native tensorflow data set format.
    Uses that format's utilities to apply preprocessing ensuring image data is compatible with model.'''
    
    # convert to tensorflow native format
    images_labels_tf = ds.from_tensor_slices((images,labels))
    
    # Apply the normalize_img function on all train and validation data and create batches
    images_labels_tf_processed = images_labels_tf.map(lambda image, label: normalize_img(image, label, img_size, n_classes))
    
    # If your data is already batched (eg, when using the image_dataset_from_directory function), remove .batch(batch_size)
    batch_generator = images_labels_tf_processed.batch(batch_size).repeat()
    
    return batch_generator
