#!/usr/bin/python3

from dataset_functions import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers
#import numpy as np

'''
Created using Python 3.7.5.
USAGE: ./classification_model.py, though these functions are intended to be imported and used elsewhere
       import via: from directory_name.file_name import function_name
	   such as:    from classification_model import *
                   from classification_model import read_dataset_into_memory, dirs
'''

'''
Builds the model to be used for sentiment classification.
	inputs:
      shape_length: int, shape of the input Dense layer
   return:
      model: keras model, the model compiled
'''
def build_sentiment_model(shape_length: int) -> Model:
   # Initialize the model layers.
   model = keras.Sequential([
      layers.Dense(160, activation='softmax', input_shape=[shape_length]),
      layers.Dense(160, activation='softmax'),
      layers.Dense(1)
	])
	
   # Compile the model.
   #    Optimizer: Stochastic gradient descent and momentum optimizer, using some standard default parameters
   #    Loss function: MSE
   #    Metrics: accuracy
   #    Note: Nesterov of True allows the momentum to be used in evaluation the steps.
   sgd_optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
   model.compile(optimizer=sgd_optimizer, loss='mse', metrics=['accuracy'])
	
   return model