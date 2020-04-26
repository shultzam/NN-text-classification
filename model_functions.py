#!/usr/bin/python3

import os
import tensorflow as tf
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
      - inputTokenCount: length of maximum input
      - lengthOfLabels: number of possible values for the labels
   return:
      - model: keras model, the model compiled
'''
def build_model(inputTokenCount: int, lengthOfLabels: int) -> Model:
   print('Building keras Sequential model with length of input: {}, labels length: {}'.format(inputTokenCount, 
                                                                                              lengthOfLabels))
   # The embedding layer which creates the weight matrix of (vocab_size) x (embedding dimension) and then indexes it. 
   # Note that in read_dataset_into_memory() we only Tokenized the 2000 most common words so the embedding layer will 
   # take that into account.
      
   # Initialize and compile the model.
   #model = Model(sequenceInput, preds)
   model = tf.keras.Sequential([
      layers.Input(shape=(inputTokenCount,), dtype="int32"),
      layers.Embedding(input_dim=2000, output_dim=512, input_length=inputTokenCount),
      layers.Dropout(0.2),
      layers.Conv1D(filters=512, kernel_size=3, padding='valid', activation='relu', strides=1),
      layers.GlobalMaxPooling1D(),
      layers.Dropout(0.2),
      layers.Dense(units=256, activation='relu'),
      layers.Dense(lengthOfLabels, activation='sigmoid')
   ])
   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	
   return model

'''
Saves files associated with the model. 
   inputs:
      - modelObject: object of the model to be saved
   return:
      - None
'''
def save_model_files(modelObject: Model):
   # Create the models directory if neede.
   modelsDir = os.path.join(os.getcwd(), 'models')
   if not os.path.isdir(modelsDir):
      os.makedirs(modelsDir)
      
   # Create the file paths.
   modelFilePath = os.path.join(modelsDir, 'model.h5')
   imageFilePath = os.path.join(modelsDir, 'model.png')
   
   # Save the model h5 file as well as a Keras representation of that file in a png.
   modelObject.save(modelFilePath)
   plot_model(modelObject, to_file=imageFilePath, show_shapes=True, show_layer_names=True)
   print()
   print('Saved model to {} and {}'.format(os.path.basename(modelFilePath), os.path.basename(imageFilePath)))
   print()   
