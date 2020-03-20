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
Created using Python 3.7.5, TensorFlow 2.0.0 and keras 2.2.4.
USAGE: ./dataset_functions.py, though these functions are intended to be imported and used elsewhere
       import via: from directory_name.file_name import function_name
	    such as:    from dataset_functions import *
                   from dataset_functions import read_dataset_into_memory, dirs
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
   SENTIMENT_NEG = 0
   SENTIMENT_POS = 1
   SENTIMENT_UNK = 2
      
''' Enumerator used for category identification. '''
class ReviewCategory(Enum):
   CAT_AMAZON = 0
   CAT_IMDB = 1
   CAT_YELP = 2
   CAT_UNKNOWN = 3
   
''' C-struct like dataclass used to store the review and their category/sentiment. '''
class Review:
   text: str
   sentiment: ReviewSentiment
   category: ReviewCategory
      
   def __init__(self, text: str, sentiment: ReviewSentiment, category: ReviewCategory):
      self.text = text
      self.sentiment = sentiment
      self.category = category


'''
Reads the dataset into memory. Stores the dataset as a list of the dataclass Review before converting into something
Tensorflow will play nice with.

   Review format: sentence \t score \n
   Returns: dataset (6 lists) and maxReviewLength (for model creation)
      6 lists:
      - reviews train, reviews test
      - sentiments train, sentiments test
      - categories train, categories test
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
         
      # Determine the review category based on the file name.
      reviewCategory = ReviewCategory.CAT_UNKNOWN
      if REVIEWS_AMAZON in dataFile:
         reviewCategory = ReviewCategory.CAT_AMAZON
      elif REVIEWS_IMDB in dataFile:
         reviewCategory = ReviewCategory.CAT_IMDB
      elif REVIEWS_YELP in dataFile:
         reviewCategory = ReviewCategory.CAT_YELP
      else:
         print('ERROR - unexpected dataset file {}. Exiting.'.format(dataFile))
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
            reviewSentiment = ReviewSentiment.SENTIMENT_UNK
            if 0 == int(splitLine[1]):
               reviewSentiment = ReviewSentiment.SENTIMENT_NEG
            elif 1 == int(splitLine[1]):
               reviewSentiment = ReviewSentiment.SENTIMENT_POS
            else:
               print('ERROR - unexpected review sentiment for review {} in file {}. Exiting.'.format(line, dataFile))
               exit()
               
            # The text portion of the review will have a trailing extra space so remove it.
            reviewText = splitLine[0].rstrip()
            
            # Update the maxReviewLength if necessary.
            currentReviewLength = len(reviewText)
            if currentReviewLength > maxReviewLength:
               maxReviewLength = currentReviewLength
            
            # Organize the review text, given sentiment and review category into a Review dataclass.
            review = Review(reviewText, reviewSentiment, reviewCategory)
            
            # Append the review dataclass to the list.
            reviewList.append(review)
            
   # Shuffle the review list.
   shuffle(reviewList)
            
   # Aggregate the three fields in to seperate lists. This is only being done to play nice with TensorFlow.
   textList = []
   sentimentList = []
   categoryList = []
   for review in reviewList:
      textList.append(review.text)
      sentimentList.append(review.sentiment.value)
      categoryList.append(review.category.value)
      
   # TODO: explain
   textTokenizer = Tokenizer(num_words=10000)
   textTokenizer.fit_on_texts(textList)
   wordIndex = textTokenizer.word_index
   textSequences = textTokenizer.texts_to_sequences(textList)
   
   # TODO: explain
   textData = pad_sequences(textSequences, maxlen=maxReviewLength)
   sentimentLabels = tf.keras.utils.to_categorical(np.array(sentimentList))
   categoryLabels = tf.keras.utils.to_categorical(np.array(categoryList))
   
   # Print shape of lists.
   print('')
   print('reviews list shape   : {}'.format(textData.shape))
   print('sentiments list shape: {}'.format(sentimentLabels.shape))
   print('categories list shape: {}'.format(categoryLabels.shape))
   print('')
   
   # Split the reviews, sentiments and categories lists into 2400:600 (80:20) ratios for train:test.
   # NOTE: this line is hilariously long but there is not really a nice way to split it. Sorry.
   reviews_train, reviews_test, sentiments_train, sentiments_test, categories_train, categories_test = train_test_split(textData, 
                                                                                                                        sentimentLabels, 
                                                                                                                        categoryLabels,
                                                                                                                        test_size = 0.2,
                                                                                                                        shuffle=False)
   
   # Package the dataset for returning.
   dataset = [reviews_train, reviews_test, sentiments_train, sentiments_test, categories_train, categories_test]
   
   return dataset, maxReviewLength
