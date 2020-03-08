#!/usr/bin/python3

from os import path
from sys import exit
from enum import Enum
from dataclasses import dataclass

'''
Created using Python 3.7.5.
USAGE: ./dataset_functions.py, though these functions are intended to be imported and used elsewhere
       import via: from directory_name.file_name import function_name
	    such as:    from dataset_functions.py import *
                   from dataset_functions.py import read_dataset_into_memory, dirs
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
@dataclass
class Review:
   text: str
   sentiment: ReviewSentiment
   category: ReviewCategory
      
   def __init__(self, text: str, sentiment: ReviewSentiment, category: ReviewCategory):
      self.text = text
      self.sentiment = sentiment
      self.category = category
   

'''
Reads the dataset into memory. Stores the dataset as a list of the dataclass Review.
Review format: sentence \t score \n
'''
def read_dataset_into_memory() -> list:
   # Initialize a list. 
   reviewList = []
   
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
            
            # Organize the review text, given sentiment and review category into a Review dataclass.
            review = Review(splitLine[0], reviewSentiment, reviewCategory)
            
            # Append the review dataclass to the list.
            reviewList.append(review)
   
   return reviewList
