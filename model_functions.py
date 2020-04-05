#!/usr/bin/python3

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers

# If this fails, Try installing graphviz via 'apt-get install graphviz' and the packages listed in requirements.txt 
# via 'pip3 install -r requirements.txt'.
from tensorflow.keras.utils import plot_model

'''
Created using Python 3.7.5, TensorFlow 2.0.1 and keras 2.2.4.
USAGE: ./model_functions.py, though these functions are intended to be imported and used elsewhere
       import via: from directory_name.file_name import function_name
	    such as:    from model_functions import *
                   from model_functionss import build_model
'''

'''
Builds the model to be used for sentiment classification. 
	inputs:
      - intputTokenCount: length of maximum input
      - lengthOfLabels: number of possible values for the labels
   return:
      - model: keras model, the model compiled
'''
def build_model(intputTokenCount: int, lengthOfLabels: int) -> Model:
   print('Building keras Sequential model with length of input: {}, labels length: {}'.format(intputTokenCount, 
                                                                                              lengthOfLabels))
   # Build the embedding layer which creates the weight matrix of (vocab_size) x (embedding dimension) and then indexes
   # this weight matrix. Note that in read_dataset_into_memory() we only Tokenized the 5000 most common words so the 
   # embedding layer shall take that into account.
   embeddingLayer = layers.Embedding(input_dim=5000, output_dim=128, input_length=intputTokenCount)
   sequenceInput = layers.Input(shape=(intputTokenCount,), dtype="int32")
   embeddedSequences = embeddingLayer(sequenceInput)
   
   # Build the model.
   # TODO: explain
   layerInstance = layers.Conv1D(128, 1, activation='relu')(embeddedSequences)
   #layerInstance = layers.MaxPooling1D(1)(layerInstance)
   layerInstance = layers.Conv1D(128, 1, activation='relu')(layerInstance)
   #layerInstance = layers.MaxPooling1D(1)(layerInstance)
   layerInstance = layers.Conv1D(128, 1, activation='relu')(layerInstance)
   
   layerInstance = layers.LSTM(64, dropout=0.2)(layerInstance)
   #layerInstance = tf.expand_dims(layerInstance, axis=-1)
   #layerInstance = layers.MaxPooling1D(32)(layerInstance)
   #layerInstance = layers.Flatten()(layerInstance)
   layerInstance = layers.Dense(128, activation='relu')(layerInstance) 
   preds = layers.Dense(lengthOfLabels, activation='softmax')(layerInstance)

   model = Model(sequenceInput, preds)

   # Compile the model based on the output shape.
   if 2 == lengthOfLabels:
      lossFunction = 'binary_crossentropy'
   else:
      lossFunction = 'categorical_crossentropy'
   model.compile(loss=lossFunction, optimizer='adam', metrics=['accuracy'])
	
   return model

'''
Saves files associated with the model. 
   inputs:
      - modelObject: object of the model to be saved
      - modelName: string identifying the name of the model
   return:
      - None
'''
def save_model_files(modelObject: Model, modelName: str):
   # Create the models directory if neede.
   modelsDir = os.path.join(os.getcwd(), 'models')
   if not os.path.isdir(modelsDir):
      os.makedirs(modelsDir)
      
   # Create the file paths.
   modelFilePath = os.path.join(modelsDir, modelName + '_model.h5')
   imageFilePath = os.path.join(modelsDir, modelName + '_model.png')
   
   # Save the model h5 file as well as a Keras representation of that file in a png.
   modelObject.save(modelFilePath)
   plot_model(modelObject, to_file=imageFilePath, show_shapes=True, show_layer_names=True)
   print()
   print('Saved model to {} and {}'.format(os.path.basename(modelFilePath), os.path.basename(imageFilePath)))
   print()   
