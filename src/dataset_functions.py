#!/usr/bin/python3

import os
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
DATA_FILES = [os.path.join(os.getcwd(), '..', REVIEW_DIR, REVIEWS_AMAZON), 
              os.path.join(os.getcwd(), '..', REVIEW_DIR, REVIEWS_IMDB), 
              os.path.join(os.getcwd(), '..', REVIEW_DIR, REVIEWS_YELP)]
PREDICTION_SENTENCES = os.path.join(os.getcwd(), '..', 'prediction_sentences', 'prediction_sentences.txt')
      
''' Enumerator used for sentiment identification. '''
class ReviewSentiment(Enum):
   SENTIMENT_NEGATIVE = 0
   SENTIMENT_POSITIVE = 1
   
''' C-struct like class used to store the review and their sentiment. '''
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
   
   Returns: 
      - dataset (4 lists):
         - dataset[0]: reviews train
         - dataset[1]: reviews test
         - dataset[2]: sentiments train
         - dataset[3]: sentiments test
      - maxTokensAllowed 
         * for model creation
      - validationReviewsText
         * for use in main.py for predictions
      - sentenceData -
         - serialized and 
         -
         * for use in main.py for predictions
'''
def read_dataset_into_memory():
   # Initialize a list for the reviews. 
   reviewList = []
   
   # Loop through each review file.
   for dataFile in DATA_FILES:
      # Verify that the review files exist.
      if not os.path.isfile(dataFile):
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
            reviewText = splitLine[0].lstrip().rstrip()
            
            # Organize the review text and review sentiment into a Review dataclass.
            review = Review(reviewText, reviewSentiment)
            
            # Append the review dataclass to the list.
            reviewList.append(review)
            
   # Shuffle the review list.
   #shuffle(reviewList)
            
   # Aggregate the fields in to two seperate lists. This is only being done to play nice with TensorFlow.
   textList = []
   sentimentList = []
   for review in reviewList:
      textList.append(review.text)
      sentimentList.append(review.sentiment.value)
   
   # Create a tokenizer based on the 2000 most common words in review texts. Then sequence it.
   textTokenizer = Tokenizer(num_words=2000, lower=True)
   textTokenizer.fit_on_texts(textList)
   textSequences = textTokenizer.texts_to_sequences(textList)
   
   # Determine the average token count amongst the reviews so that shorter can be padded and longer
   # can be truncated,
   tokensEach = [len(tokens) for tokens in textSequences]
   avgTokens = sum(tokensEach) / len(tokensEach)
   maxTokens = int(avgTokens * 1.75)
 
   # Pad sequences so that each review is the same 'length' in tokens.
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
   
   # Convert the reviews_test list back to text for prediction use in main.py.
   validateReviewsText = textTokenizer.sequences_to_texts(reviews_test)
   
   # Package the dataset for returning.
   dataset = [reviews_train, reviews_test, sentiments_train, sentiments_test]
   
   # Verify that the prediction sentences files exist.
   if not os.path.isfile(PREDICTION_SENTENCES):
      print('ERROR - dataset file {} not found. Exiting.'.format(PREDICTION_SENTENCES))
      exit()
      
   # Read the sentences into memory.
   predictionList = []
   print('Reading prediction sentences file {} into memory..'.format(PREDICTION_SENTENCES))
   with open(PREDICTION_SENTENCES) as dataFileObj:
      lines = []
      for line in dataFileObj:
         # Skip over comments in the file.
         if line[0] == '#':
            continue
         
         # Remove the newline from the sentence.
         sentenceText = line.replace('\n', '')

         # The text portion of the sentence will have a trailing extra space so remove it.
         sentenceText = sentenceText.rstrip()

         # Append the sentence dataclass to the list.
         predictionList.append(sentenceText)
   
   # Create a tokenizer based on the 2000 most common words in review texts. Then sequence it. 
   # Note that the sentimentless sentences are sequenced using the sequence fitting from the reviews.
   predictionSequences = textTokenizer.texts_to_sequences(predictionList)
 
   # Pad sequences so that each sentence is the same 'length' in tokens. Same token count as the reviews so that 
   # predictions are possible.
   paddedSentenceSequences = pad_sequences(sequences=predictionSequences, maxlen=maxTokens, padding='post')
   
   # Package the sentences data for returning.
   sentenceData = [paddedSentenceSequences, predictionList]
   
   return dataset, maxTokens, validateReviewsText, sentenceData

'''
Gathers some reviews that the model was very correct, very incorrect or very confused about.
   inputs:
      - predictionList: list of prediction weights from model.predict() 
         -in format [[negativeConfidence0, positiveConfidence0], negativeConfidenceN, positiveConfidenceN]
      - reviewTexts: serialized review strings 
         - in format [[review0], .. [reviewN]]
      - actualSentiments: list of actual review sentiments 
         - in format [[0.0, 1.0], .. [1.0, 0.0]]
   returns:
      - 3 lists:
         - reviewsList, predictionsList, actualsList
         - formatted like this for a pretty table formatting
'''
def gather_interesting_reviews(predictionList: list, reviewTexts: list, actualSentiments: list):
   # Initialize the list.
   reviewsList = []
   predictionsList = []
   actualsList = []
   
   # Gather the correctList items.
   actualPositiveCount = 0
   actualNegativeCount = 0
   for index in range (0, len(predictionList)):
      # Locally save some fields.
      prediction = predictionList[index]
      actual = actualSentiments[index]

      # If the sentiment was correct and the model was super sure, save this review.
      if (prediction[0] > 0.99) and (actual[0] == 1.0) and (actualNegativeCount < 5):
         reviewsList.append(reviewTexts[index])
         predictionsList.append(prediction)
         actualsList.append(False)
         actualNegativeCount += 1
         continue
      elif (prediction[1] > 0.99) and (actual[1] == 1.0) and (actualPositiveCount < 5):
         reviewsList.append(reviewTexts[index])
         predictionsList.append(prediction)
         actualsList.append(True)
         actualPositiveCount += 1
         continue

      # If 10 correct items were found, break.
      if (5 == actualNegativeCount and 5 == actualNegativeCount):
         break
         
   # Gather the incorrectList items.
   actualPositiveCount = 0
   actualNegativeCount = 0
   for index in range (0, len(predictionList)):
      # Locally save some fields.
      prediction = predictionList[index]
      actual = actualSentiments[index]

      # If the sentiment was incorrect and the model was super sure, save this review.
      if (prediction[0] > 0.99) and (actual[1] == 1.0) and (actualPositiveCount < 5):
         reviewsList.append(reviewTexts[index])
         predictionsList.append(prediction)
         actualsList.append(True)
         actualPositiveCount += 1
         continue
      elif (prediction[1] > 0.99) and (actual[0] == 1.0) and (actualNegativeCount < 5):
         reviewsList.append(reviewTexts[index])
         predictionsList.append(prediction)
         actualsList.append(False)
         actualNegativeCount += 1
         continue

      # If 10 incorrect items were found, break.
      if (5 == actualNegativeCount and 5 == actualNegativeCount):
         break
         
   # Gather the confusedList items.
   count = 0
   for index in range (0, len(predictionList)):
      # Locally save some fields.
      prediction = predictionList[index]
      actual = actualSentiments[index]

      # If the sentiment was false and the model was super sure, save this review.
      if (0.3 <= prediction[0] <= 0.7) and (count < 10):
         if actual[0] == 1.0:
            reviewsList.append(reviewTexts[index])
            predictionsList.append(prediction)
            actualsList.append(False)
            count += 1
         elif actual[1] == 1.0 and (count < 10):
            reviewsList.append(reviewTexts[index])
            predictionsList.append(prediction)
            actualsList.append(True)
            count += 1
         continue

      # If 10 correct items were found, break.
      if (10 == count):
         break
         
   # Return the lists.
   return reviewsList, predictionsList, actualsList
