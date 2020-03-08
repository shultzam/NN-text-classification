#!/usr/bin/python3

import os
from dataset_functions import *
from classification_model import *

'''
Created using Python 3.7.5.
USAGE: ./main.py
'''

# Silence a TensorFlow warning stating that TensorFlow isn't optimized for our CPU. It's fine, we're running on a GPU.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Read the dataset into memory. The following indicies rules apply:
#    dataset[0] - list containing review texts
#    dataset[1] - list containing review sentiments
#    dataset[2] - list containing review categories
dataset = read_dataset_into_memory()

# Initialize the model.
model = build_sentiment_model(len(dataset))