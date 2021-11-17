# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 20:20:41 2021

@author: bettmensch
"""

# imports
import os
os.chdir('C:/Users/bettmensch/GitReps/fashion_app/ml_model_development/code')
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import pandas as pd
from question_1_utils import load_mnist, preprocess_data
from matplotlib import pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

mnist_repo = os.path.join('..','..','fashion-mnist')
mnist_data_dir = os.path.join(mnist_repo,'data','fashion')

# -- [1] Get data
# load meta data
labels_mapping_df = pd.read_csv(os.path.join('..','meta','target_label_mapping.csv'))

# load data
images_train, labels_train = load_mnist(mnist_data_dir,
                                        'train')

images_test, labels_test = load_mnist(mnist_data_dir,
                                        't10k')

# --- [2] EDA
labels_train_df = pd.DataFrame(data = labels_train,
                               columns=['index']). \
                    merge(labels_mapping_df,
                          how='left',
                          on='index')

labels_test_df = pd.DataFrame(data = labels_test,
                               columns=['index']). \
                    merge(labels_mapping_df,
                          how='left',
                          on='index')

# check target distribution
labels_train_df['label'].value_counts() # perfectly balanced - as all things should be
labels_test_df['label'].value_counts() # matches training data distribution - no reweighting of rare classes will be needed

def plot_observation(sample_index):
    '''Plots an individual observation, i.e. image, in gray scale with its target label'''
    sample_label = labels_train_df.iloc[sample_index].to_dict()['label']
    sample_image = images_train[sample_index].reshape((28,28))
    
    sample_image = sample_image.reshape(28,28)
    plt.title('Label is {label}'.format(label=sample_label))
    plt.imshow(sample_image, cmap='gray')
    plt.show()

# plot some obs
for sample_index in range(10):
    plot_observation(sample_index)
    
# --- [3] Preprocessing data

# reshape images to tensors [n_obs, width, height, n_channel]
images_train_reshaped = np.expand_dims(images_train.reshape(images_train.shape[0],28,28),axis=-1)
images_test_reshaped = np.expand_dims(images_test.reshape(images_test.shape[0],28,28),axis=-1)

# create native tensorflow batch generators
# NOTE: I could avoid these shenanigans by simply using numpy arrays, but you guys mentioned that you had your own image data sets,
# so this preprocessing using the tensorflow data class might be useful
batch_size = 200
num_classes = len(set(labels_train))
num_train_batches = int(len(labels_train)/batch_size)
num_test_batches = int(len(labels_test)/batch_size)

data_train_tf = preprocess_data(images=images_train_reshaped, 
                                  labels=labels_train,
                                  batch_size=batch_size)

data_test_tf = preprocess_data(images=images_test_reshaped,
                                 labels=labels_test,
                                 batch_size=batch_size)

# =============================================================================
# images_train_reshaped = images_train_reshaped/255
# images_test_reshaped = images_test_reshaped/255
# =============================================================================

# --- [1] build model

# define model
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', input_shape=[28,28,1])) # any other data used to train set will have to have this shape
model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# define loss and optimizer
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# early stopping w.r.t validate
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=3)

model.fit(#images_train_reshaped,
          #labels_train,
          data_train_tf,
          verbose=1,
          epochs=15,
          steps_per_epoch=num_train_batches,
          #validation_split=0.2,
          validation_data=data_test_tf,
          validation_steps=num_test_batches,
          callbacks=[es])

# =============================================================================
# should give something along these lines (model weights etc arent seeded, so results may vary):
#     Epoch 1/15
#     300/300 [==============================] - 11s 35ms/step - loss: 0.5054 - accuracy: 0.8207 - val_loss: 0.3593 - val_accuracy: 0.8731
#     Epoch 2/15
#     300/300 [==============================] - 11s 35ms/step - loss: 0.3336 - accuracy: 0.8808 - val_loss: 0.3052 - val_accuracy: 0.8894
#     Epoch 3/15
#     300/300 [==============================] - 11s 36ms/step - loss: 0.2825 - accuracy: 0.8962 - val_loss: 0.2883 - val_accuracy: 0.8927
#     Epoch 4/15
#     300/300 [==============================] - 11s 38ms/step - loss: 0.2543 - accuracy: 0.9081 - val_loss: 0.2711 - val_accuracy: 0.9015
#     Epoch 5/15
#     300/300 [==============================] - 12s 39ms/step - loss: 0.2323 - accuracy: 0.9155 - val_loss: 0.2591 - val_accuracy: 0.9054
#     Epoch 6/15
#     300/300 [==============================] - 11s 36ms/step - loss: 0.2134 - accuracy: 0.9217 - val_loss: 0.2542 - val_accuracy: 0.9071
#     Epoch 7/15
#     300/300 [==============================] - 10s 33ms/step - loss: 0.1990 - accuracy: 0.9271 - val_loss: 0.2494 - val_accuracy: 0.9109
#     Epoch 8/15
#     300/300 [==============================] - 10s 34ms/step - loss: 0.1864 - accuracy: 0.9321 - val_loss: 0.2451 - val_accuracy: 0.9129
#     Epoch 9/15
#     300/300 [==============================] - 10s 34ms/step - loss: 0.1720 - accuracy: 0.9368 - val_loss: 0.2531 - val_accuracy: 0.9114
#     Epoch 10/15
#     300/300 [==============================] - 11s 37ms/step - loss: 0.1636 - accuracy: 0.9400 - val_loss: 0.2488 - val_accuracy: 0.9146
#     Epoch 11/15
#     300/300 [==============================] - 11s 36ms/step - loss: 0.1512 - accuracy: 0.9450 - val_loss: 0.2555 - val_accuracy: 0.9120
#     Epoch 12/15
#     300/300 [==============================] - 10s 33ms/step - loss: 0.1423 - accuracy: 0.9479 - val_loss: 0.2597 - val_accuracy: 0.9135
#     Epoch 13/15
#     300/300 [==============================] - 12s 39ms/step - loss: 0.1369 - accuracy: 0.9490 - val_loss: 0.2601 - val_accuracy: 0.9128
#     Epoch 00013: early stopping
#     INFO:tensorflow:Assets written to: ..\model\cnn_model\assets
#     Accuracy on the train set is:  0.9612666666666667
#     Accuracy on the test set is:  0.9128
# -> 2ppts above fashion mnist benchmarks found here: http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/
# =============================================================================
    
# export trained model
model_export_dir = os.path.join('..','artifacts')

if not os.path.exists(model_export_dir):
    os.makedirs(model_export_dir)
    
model.save(os.path.join(model_export_dir,'cnn_model'))

# export optimization results
optimization_metrics_df = pd.DataFrame(model.history.history)
optimization_metrics_df.to_csv(os.path.join(model_export_dir,'results.csv'))

# --- [2] evaluate model
# get train predictions
label_prob_cols = [label for label in labels_mapping_df['label'].tolist()]

p_train = model.predict(data_train_tf,steps=num_train_batches)
p_train_df = pd.DataFrame(p_train,
                          columns = label_prob_cols)
p_train_df['response'] = labels_train_df['label']
p_train_df['prediction_index'] = np.where(p_train==np.max(p_train,axis=1).reshape(-1,1))[1]
p_train_df = p_train_df.merge(labels_mapping_df,
                              left_on = 'prediction_index',
                              right_on = 'index',
                              how='left'). \
                        rename(columns={'label':'prediction'}). \
                        drop('index',axis=1)
                        
p_train_df.to_csv(os.path.join(model_export_dir,'p_train.csv'))

# get test predictions
p_test = model.predict(data_test_tf,steps=num_test_batches)
p_test_df = pd.DataFrame(p_test,
                         columns = label_prob_cols)
p_test_df['response'] = labels_test_df['label']
p_test_df['prediction_index'] = np.where(p_test==np.max(p_test,axis=1).reshape(-1,1))[1]
p_test_df = p_test_df.merge(labels_mapping_df,
                              left_on = 'prediction_index',
                              right_on = 'index',
                              how='left'). \
                        rename(columns={'label':'prediction'}). \
                        drop('index',axis=1)
                        
p_test_df.to_csv(os.path.join(model_export_dir,'p_test.csv'))
                        
# evaluate on train
train_acc = (p_train_df['response'] == p_train_df['prediction']).mean()
print('Accuracy on the train set is: ', train_acc)

conf_train = p_train_df[['response','prediction']]. \
                groupby(['response','prediction']). \
                agg(len). \
                reset_index(). \
                pivot(index='response',columns='prediction',values=0). \
                fillna(0)
                
conf_train.to_csv(os.path.join(model_export_dir,'confusion_train.csv'))

# evaluate on test
test_acc = (p_test_df['response'] == p_test_df['prediction']).mean()
print('Accuracy on the test set is: ', test_acc)

conf_test = p_test_df[['response','prediction']]. \
                groupby(['response','prediction']). \
                agg(len). \
                reset_index(). \
                pivot(index='response',columns='prediction',values=0). \
                fillna(0)
                
conf_test.to_csv(os.path.join(model_export_dir,'confusion_test.csv'))
                
# we can see some similar clothes items understandably causing confusion:
# 1. Shirt vs T-shirt/top
# 2. Shirt vs Pullover
# 3. Pullover vs Coat etc