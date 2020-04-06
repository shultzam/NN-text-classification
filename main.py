#!/usr/bin/python3

import os
import sys
from dataset_functions import *
from model_functions import *

'''
Created using Python 3.7.5, TensorFlow 2.0.1 and keras 2.2.4.
USAGE: ./main.py
'''

# Determine if model files are to be saved.
SAVE_MODELS = False
if len(sys.argv) > 1:
   if (sys.argv[1]).lower() == 'save':
      SAVE_MODELS = True

#####################
# Dataset functionality.
#####################

# Print library versions in use.
print('Tensorflow version: {}, keras version: {}'.format(tf.__version__, tf.keras.__version__))
print()

# Silence a TensorFlow warning stating that TensorFlow isn't optimized for our CPU. It's fine-ish.
# We could configure to run on the GPU but this process doesn't take so long that it is absolutely needed.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Read the dataset into memory. The following indicies rules apply to the returned dataset:
#    dataset[0] - list containing review texts (training set)
#    dataset[1] - list containing review texts (test set)
#    dataset[2] - list containing review categories as numpy categoricals (training set)
#    dataset[3] - list containing review categories as numpy categoricals (test set)
dataset, maxTokens = read_dataset_into_memory()

#####################
# Train, test and evaluate the model.
#####################

# Initialize the sentiment model using the maximum review length and the number of potential outputs.
model = build_model(maxTokens, dataset[2].shape[1])

# Print a summary of the sentiment model.
model.summary()

# Fit the sentiment model and test it against the test data.
#   - epochs    : the number of iterations over the entire x and y datas to train the model.
#   - batch_size: the number of samples per gradient update.
history = model.fit(x=dataset[0], y=dataset[2], validation_split=0.2, epochs=20, batch_size=128)
#print('TEMP | history: \n{}'.format(history.history))

# Determine the accuracy of the sentiment model.
result = model.evaluate(dataset[1], dataset[3], verbose=0)
print()
print("Sentiment model accuracy: {0:.2%}".format(result[1]))

# Save the sentiment model if deemed necessary.
if SAVE_MODELS:
   save_model_files(model)

# TODO: create plots of training info.
