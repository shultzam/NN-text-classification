#!/usr/bin/python3

import os
import sys
from dataset_functions import *
from model_functions import *
from tensorflow.keras import backend as kback
import numpy as np

'''
Created using Python 3.7.5, TensorFlow 2.0.1 and keras 2.2.4.
USAGE: ./main.py
'''

#####################
# Parse commandline argument to determine if the model or plots are created and saved.
#####################
SAVE_MODEL = False
SAVE_PLOT  = False
for arg in sys.argv:
   if 'model' == arg.lower():
      SAVE_MODEL = True
   elif 'plot' == arg.lower():
      SAVE_PLOT = True
   elif arg != sys.argv[0]:
      print('WARNING - unrecognized argument: {}'.format(arg))

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
#    dataset[1] - list containing review texts (validate set)
#    dataset[2] - list containing review categories as numpy categoricals (training set)
#    dataset[3] - list containing review categories as numpy categoricals (validate set)
dataset, maxTokens, validateReviewsText, sentenceData = read_dataset_into_memory()

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
history = model.fit(x=dataset[0], y=dataset[2], validation_split=0.2, epochs=30, batch_size=128)

# Determine the accuracy of the sentiment model. loss is stored  at [0].
accuracy = model.evaluate(dataset[1], dataset[3], verbose=0)[1]
print('\nModel accuracy: {0:.2%}'.format(accuracy))

#####################
# Use the model to make predictions on the validation set.
#####################

# Allow the model to predict against each input.
predictions = model.predict(dataset[1])

# Collect the predictions for 10 reviews that were very correct, very incorrect and very confusing to the model.
# Split down the middle if possible for positive/negative sentiment.
correctList, incorrectList, confusingList = gather_interesting_reviews(predictions,
                                                                       validateReviewsText,
                                                                       dataset[3])

print('\n{} correct predictions:'.format(len(correctList)))
for item in correctList:
   print('   {}, {}, {}'.format(item.text, item.confidence, item.sentiment))
   
print('\n{} incorrect predictions:'.format(len(incorrectList)))
for item in incorrectList:
   print('   {}, {}, {}'.format(item.text, item.confidence, item.sentiment))

print('\n{} confused predictions:'.format(len(confusingList)))
for item in confusingList:
   print('   {}, {}, {}'.format(item.text, item.confidence, item.sentiment))

#####################
# Use the model to make predictions on a set of reviews carefully crafted to lack any sentiment.
#####################

# Allow the model to predict against each input.
predictions = model.predict(sentenceData[0])
print('\nsentimentless predictions:')
for index in range(0, len(predictions)):
   print('   {}: {}'.format(sentenceData[1][index], predictions[index]))

#####################
# If prompted, save the model image and plot.
#####################

# Save the model if prompted.
if SAVE_MODEL:
   save_model_files(model)

# Save the plot if prompted.
if SAVE_PLOT:
   save_plot_file(history.history, accuracy)
