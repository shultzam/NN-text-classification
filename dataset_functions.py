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

''' Enumerator used for source identification. '''
class ReviewSource(Enum):
   SOURCE_AMAZON = 0
   SOURCE_IMDB = 1
   SOURCE_YELP = 2
   SOURCE_UNKNOWN = 3
      
''' Enumerator used for category identification, representing source of review and sentiment. '''
class ReviewCategory(Enum):
   CAT_AMAZON_NEG = 0
   CAT_AMAZON_POS = 1
   CAT_IMDB_NEG = 2
   CAT_IMDB_POS = 3
   CAT_YELP_NEG = 4
   CAT_YELP_POS = 5
   CAT_UNKNOWN = 6
   
''' C-struct like dataclass used to store the review and their category/sentiment. '''
class Review:
   text: str
   category: ReviewCategory
      
   def __init__(self, text: str, category: ReviewCategory):
      self.text = text
      self.category = category


'''
Reads the dataset into memory. Stores the dataset as a list of the dataclass Review before converting into something
Tensorflow will play nice with.

   Review format: sentence \t score \n
   Returns: dataset (4 lists) and maxTokensAllowed (for model creation)
      4 lists:
      - reviews train, reviews test
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
      reviewSource = ReviewSource.SOURCE_UNKNOWN
      if REVIEWS_AMAZON in dataFile:
         reviewSource = ReviewSource.SOURCE_AMAZON
      elif REVIEWS_IMDB in dataFile:
         reviewSource = ReviewSource.SOURCE_IMDB
      elif REVIEWS_YELP in dataFile:
         reviewSource = ReviewSource.SOURCE_YELP
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
            reviewCategory = determine_review_category(reviewSource, splitLine)
               
            # The text portion of the review will have a trailing extra space so remove it.
            reviewText = splitLine[0].rstrip()
            
            # Update the maxReviewLength if necessary.
            currentReviewLength = len(reviewText)
            if currentReviewLength > maxReviewLength:
               maxReviewLength = currentReviewLength
            
            # Organize the review text, given sentiment and review category into a Review dataclass.
            review = Review(reviewText, reviewCategory)
            
            # Append the review dataclass to the list.
            reviewList.append(review)
            
   # Shuffle the review list.
   shuffle(reviewList)
            
   # Aggregate the fields in to two seperate lists. This is only being done to play nice with TensorFlow.
   textList = []
   categoryList = []
   for review in reviewList:
      textList.append(review.text)
      categoryList.append(review.category.value)
   
   # TODO: explain
   textTokenizer = Tokenizer(num_words=5000)
   textTokenizer.fit_on_texts(textList)
   wordIndex = textTokenizer.word_index
   textSequences = textTokenizer.texts_to_sequences(textList)
   
   # TODO: explain
   numTokensEach = [len(tokens) for tokens in textSequences]
   avgTokens = sum(numTokensEach) / len(numTokensEach)
   maxTokens = int(avgTokens * 1.2)
   
   # TODO: explain
   textData = pad_sequences(textSequences, maxlen=maxTokens)
   categoryLabels = tf.keras.utils.to_categorical(np.array(categoryList))
   
   # Print shape of lists.
   print('')
   print('reviews list shape   : {}'.format(textData.shape))
   print('categories list shape: {}'.format(categoryLabels.shape))
   print('')
   
   # Split the reviews, sentiments and categories lists into 2400:600 (80:20) ratios for train:test.
   reviews_train, reviews_test, categories_train, categories_test = train_test_split(textData,
                                                                                     categoryLabels,
                                                                                     test_size = 0.2,
                                                                                     shuffle=False)
   
   # Package the dataset for returning.
   dataset = [reviews_train, reviews_test, categories_train, categories_test]
   
   return dataset, maxTokens


'''
Helper function for read_dataset_into_memory. Used to determinet he proper category for the review in question.
   inputs:
      - source: ReviewSource value
      - lineTokens: tokenized review string
   return:
      - model: keras model, the model compiled
'''
def determine_review_category(source: ReviewSource, lineTokens: list) -> ReviewCategory:
   # Obtain the sentiment of the review of interest.
   reviewSentiment = int(lineTokens[1])
   
   if source == ReviewSource.SOURCE_AMAZON and 0 == reviewSentiment:
      return ReviewCategory.CAT_AMAZON_NEG
   elif source == ReviewSource.SOURCE_AMAZON and 1 == reviewSentiment:
      return ReviewCategory.CAT_AMAZON_POS
   elif source == ReviewSource.SOURCE_IMDB and 0 == reviewSentiment:
      return ReviewCategory.CAT_IMDB_NEG
   elif source == ReviewSource.SOURCE_IMDB and 1 == reviewSentiment:
      return ReviewCategory.CAT_IMDB_POS
   elif source == ReviewSource.SOURCE_YELP and 0 == reviewSentiment:
      return ReviewCategory.CAT_YELP_NEG
   elif source == ReviewSource.SOURCE_YELP and 1 == reviewSentiment:
      return ReviewCategory.CAT_YELP_POS
   
   # If none of the above events trigger then this function has failed and the program should exit.
   print('ERROR - unexpected review category source or sentiment from review: {}. Exiting.'.format(' '.join(lineTokens)))
   exit()
