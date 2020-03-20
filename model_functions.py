#!/usr/bin/python3

from dataset_functions import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers

'''
Created using Python 3.7.5, TensorFlow 2.0.0 and keras 2.2.4.
USAGE: ./model_functions.py, though these functions are intended to be imported and used elsewhere
       import via: from directory_name.file_name import function_name
	    such as:    from model_functions import *
                   from model_functionss import read_dataset_into_memory, dirs
'''

'''
Builds the model to be used for sentiment classification. 
	inputs:
      - lengthOfInput: length of maximum input
      - lengthOfLabels: number of possible values for the labels
   return:
      - model: keras model, the model compiled
'''
def build_model(lengthOfInput: int, lengthOfLabels: int) -> Model:
   print('Building keras Sequential model with length of input: {}, labels length: {}'.format(lengthOfInput, 
                                                                                              lengthOfLabels))
   # Build the embedding layer which creates the weight matrix of (vocab_size) x (embedding dimension) and then indexes
   # this weight matrix. Note that in read_dataset_into_memory() we only Tokenized the 10000 most common words so the 
   # embedding layer shall take that into account.
   embeddingLayer = layers.Embedding(input_dim=10000, output_dim=128, input_length=lengthOfInput)
   sequenceInput = layers.Input(shape=(lengthOfInput,), dtype="int32")
   embeddedSequences = embeddingLayer(sequenceInput)
   
   # Build the model.
   layerInstance = layers.Conv1D(128, 5, activation='relu')(embeddedSequences)
   layerInstance = layers.MaxPooling1D(5)(layerInstance)
   layerInstance = layers.Conv1D(128, 5, activation='relu')(layerInstance)
   layerInstance = layers.MaxPooling1D(5)(layerInstance)
   layerInstance = layers.Conv1D(128, 5, activation='relu')(layerInstance)
   
   layerInstance = layers.LSTM(64, dropout=0.2)(layerInstance)
   layerInstance = tf.expand_dims(layerInstance, axis=-1)
   layerInstance = layers.MaxPooling1D(32)(layerInstance)
   layerInstance = layers.Flatten()(layerInstance)
   layerInstance = layers.Dense(128, activation='relu')(layerInstance)
   
   # Compile the model.
   # TODO: explain
   preds = layers.Dense(lengthOfLabels, activation='softmax')(layerInstance)
   model = Model(sequenceInput, preds)
   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
   return model