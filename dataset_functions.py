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
NO_SENTIMENT_SENTENCES = path.join('sentiment_lacking_sentences', 'sentimentless_sentences.txt')
      
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

''' C-struct like class used to store the review, the prediction confidence and the actual sentiment. '''
class ReviewWithPrediction:
   text: str
   confidence: float
   sentiment: bool
      
   def __init__(self, text: str, confidence: float, sentiment: bool):
      self.text = text
      self.confidence = confidence
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
            
            # Organize the review text and review sentiment into a Review dataclass.
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
   
   # Convert the reviews_test list back to text for prediction use in main.py.
   validateReviewsText = textTokenizer.sequences_to_texts(reviews_test)
   
   # Package the dataset for returning.
   dataset = [reviews_train, reviews_test, sentiments_train, sentiments_test]
   
   # Process the sentiment-less files.
   sentimentlessList = []
   
   # Verify that the review files exist.
   if not path.isfile(NO_SENTIMENT_SENTENCES):
      print('ERROR - dataset file {} not found. Exiting.'.format(NO_SENTIMENT_SENTENCES))
      exit()
      
   # Read the sentences into memory.
   print('Reading sentimentless file {} into memory..'.format(NO_SENTIMENT_SENTENCES))
   with open(NO_SENTIMENT_SENTENCES) as dataFileObj:
      lines = []
      for line in dataFileObj:
         # Remove the newline from the sentence.
         sentenceText = line.replace('\n', '')

         # The text portion of the sentence will have a trailing extra space so remove it.
         sentenceText = sentenceText.rstrip()

         # Append the sentence dataclass to the list.
         sentimentlessList.append(sentenceText)
   
   # TODO: explain. Using the same sequence fitting from the reviews.
   sentenceSequences = textTokenizer.texts_to_sequences(sentimentlessList)
   
   # TODO: explain
   tokensEach = [len(tokens) for tokens in sentenceSequences]
   avgSentenceTokens = sum(tokensEach) / len(tokensEach)
   maxSentenceTokens = int(avgTokens * 1.5)
 
   # TODO: explain
   paddedSentenceSequences = pad_sequences(sequences=sentenceSequences, maxlen=maxSentenceTokens, padding='post')
   
   # Package the sentences data for returning.
   sentenceData = [paddedSentenceSequences, sentimentlessList]
   
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
      - correctList of ReviewWithPrediction objects
      - incorrectList ReviewWithPrediction objects
      - confusedList ReviewWithPrediction objects
'''
def gather_interesting_reviews(predictionList: list, reviewTexts: list, actualSentiments: list):
   # Initialize some lists.
   correctList = []
   incorrectList = []
   confusedList = []
   
   # Gather the correctList items.
   positiveCount = 0
   negativeCount = 0
   for index in range (0, len(predictionList)):
      # Locally save some fields.
      prediction = predictionList[index]
      actual = actualSentiments[index]

      # If the sentiment was false and the model was super sure, save this review.
      if (prediction[0] > 0.985) and (actual[0] == 1.0) and (negativeCount < 5):
         correctList.append(ReviewWithPrediction(reviewTexts[index], prediction[0], False))
         negativeCount += 1
         continue
      elif (prediction[1] > 0.985) and (actual[1] == 1.0) and (positiveCount < 5):
         correctList.append(ReviewWithPrediction(reviewTexts[index], prediction[1], False))
         positiveCount += 1
         continue

      # If 10 correct items were found, break.
      if (len(correctList) >= 10):
         break
         
   # Gather the incorrectList items.
   positiveCount = 0
   negativeCount = 0
   for index in range (0, len(predictionList)):
      # Locally save some fields.
      prediction = predictionList[index]
      actual = actualSentiments[index]

      # If the sentiment was false and the model was super sure, save this review.
      if (prediction[0] > 0.985) and (actual[1] == 1.0) and (negativeCount < 5):
         incorrectList.append(ReviewWithPrediction(reviewTexts[index], prediction[0], False))
         negativeCount += 1
         continue
      elif (prediction[1] > 0.985) and (actual[0] == 1.0) and (positiveCount < 5):
         incorrectList.append(ReviewWithPrediction(reviewTexts[index], prediction[1], False))
         positiveCount += 1
         continue

      # If 10 correct items were found, break.
      if (len(incorrectList) >= 10):
         break
         
   # Gather the confusedList items.
   positiveCount = 0
   negativeCount = 0
   for index in range (0, len(predictionList)):
      # Locally save some fields.
      prediction = predictionList[index]
      actual = actualSentiments[index]

      # If the sentiment was false and the model was super sure, save this review.
      if (0.3 <= prediction[0] <= 0.7) and (actual[1] == 1.0) and (negativeCount < 7):
         confusedList.append(ReviewWithPrediction(reviewTexts[index], prediction[0], False))
         negativeCount += 1
         continue
      elif (0.3 <= prediction[1] <= 0.7) and (actual[0] == 1.0) and (positiveCount < 7):
         confusedList.append(ReviewWithPrediction(reviewTexts[index], prediction[1], False))
         positiveCount += 1
         continue

      # If 10 correct items were found, break.
      if (len(confusedList) >= 10):
         break
         
   # Return the lists.
   return correctList, incorrectList, confusedList
