#!/usr/bin/python3

from os import path
from sys import exit
from enum import Enum
from random import shuffle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split

'''
Created using Python 3.7.5, TensorFlow 2.0.1 and keras 2.2.4.
USAGE: ./dataset_functions.py, though these functions are intended to be imported and used elsewhere
       import via: from directory_name.file_name import function_name
	    such as:    from dataset_functions import *
                   from dataset_functions import read_dataset_into_memory
'''

''' Dataset related constants '''
REVIEW_DIR = 'sentiment_labelled_sentences'
REVIEWS_AMAZON = 'amazon_cells_labelled.txt'
REVIEWS_IMDB = 'imdb_labelled.txt'
REVIEWS_YELP = 'yelp_labelled.txt'
DATA_FILES = [path.join(REVIEW_DIR, REVIEWS_AMAZON), 
              path.join(REVIEW_DIR, REVIEWS_IMDB), 
              path.join(REVIEW_DIR, REVIEWS_YELP)]
      
''' Enumerator used for sentiment identification. '''
class ReviewSentiment(Enum):
   SENTIMENT_NEGATIVE = 0
   SENTIMENT_POSITIVE = 1
   
''' C-struct like dataclass used to store the review and their sentiment. '''
class Review:
   text: str
   sentiment: ReviewSentiment
      
   def __init__(self, text: str, sentiment: ReviewSentiment):
      self.text = text
      self.sentiment = sentiment


'''
Reads the dataset into memory. Stores the dataset as a list of the dataclass Review before converting into something
Tensorflow will play nice with.

   Review format: sentence \t score \n
   Returns: dataset (4 lists) and maxTokensAllowed (for model creation)
      4 lists:
      - reviews train, reviews test
      - sentiments train, sentiments test
'''
def read_dataset_into_memory():
   # Initialize a list for the reviews. 
   reviewList = []
   
   # Track the maximum length of a review. Used later with Tensorflow operations.
   maxReviewLength = 0
   
   # Loop through each review file.
   for dataFile in DATA_FILES:
      # Verify that the review files exist.
      if not path.isfile(dataFile):
         print('ERROR - dataset file {} not found. Exiting.'.format(dataFile))
         exit()
         
      # Read each data file's reviews into memory.
      print('Reading review file {} into memory..'.format(dataFile))
      with open(dataFile) as dataFileObj:
         lines = []
         for line in dataFileObj:
            # Remove the newline from the review.
            line = line.replace('\n', '')
            
            # Split the line on the tab character and assign the sentiment to an enum.
            splitLine = line.split('\t')
            reviewSentiment = -5
            if 0 == int(splitLine[1]):
               reviewSentiment = ReviewSentiment.SENTIMENT_NEGATIVE
            elif 1 == int(splitLine[1]):
               reviewSentiment = ReviewSentiment.SENTIMENT_POSITIVE
            else:
               print('ERROR - Unknown sentiment: {}'.format(int(splitLine[1])))
               exit()
               
            # The text portion of the review will have a trailing extra space so remove it.
            reviewText = splitLine[0].rstrip()
            
            # Update the maxReviewLength if necessary.
            currentReviewLength = len(reviewText)
            if currentReviewLength > maxReviewLength:
               maxReviewLength = currentReviewLength
            
            # Organize the review text, given sentiment and review sentiment into a Review dataclass.
            review = Review(reviewText, reviewSentiment)
            
            # Append the review dataclass to the list.
            reviewList.append(review)
            
   # Shuffle the review list.
   shuffle(reviewList)
            
   # Aggregate the fields in to two seperate lists. This is only being done to play nice with TensorFlow.
   textList = []
   sentimentList = []
   for review in reviewList:
      textList.append(review.text)
      sentimentList.append(review.sentiment.value)
   
   # TODO: explain
   textTokenizer = Tokenizer(num_words=2000, lower=True)
   textTokenizer.fit_on_texts(textList)
   wordIndex = textTokenizer.word_index
   textSequences = textTokenizer.texts_to_sequences(textList)
   
   # TODO: explain
   tokensEach = [len(tokens) for tokens in textSequences]
   avgTokens = sum(tokensEach) / len(tokensEach)
   maxTokens = int(avgTokens * 1.5)
 
   # TODO: explain
   textData = pad_sequences(sequences=textSequences, maxlen=maxTokens, padding='post')
   sentimentLabels = tf.keras.utils.to_categorical(np.array(sentimentList))
   
   # Print shape of lists.
   print('')
   print('reviews list shape   : {}'.format(textData.shape))
   print('sentiments list shape: {}'.format(sentimentLabels.shape))
   print('')
   
   # Split the reviews and sentiments lists into 2400:600 (80:20) ratios for train:test.
   reviews_train, reviews_test, sentiments_train, sentiments_test = train_test_split(textData,
                                                                                     sentimentLabels,
                                                                                     test_size=0.2,
                                                                                     shuffle=False)
   
   # Package the dataset for returning.
   dataset = [reviews_train, reviews_test, sentiments_train, sentiments_test]
   
   return dataset, maxTokens
