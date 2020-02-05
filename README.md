# Preamble
Using Neural Networks in Review Classification  
Allen Shultz  
ashultz2019@my.fit.edu  
Florida Institute of Technology  
ECE 5268, Spring 2020  
repo: github.com/shultzam/NN-text-classification  

# Description
The ability to process and understand sentiment associated with text is a topic that impacts many facets in everyday life. In this case, with an opinion or review of services. Sentiment is much more subtle than a star rating system and is therefore harder to process. A neural network can help quantify sentiment behind a review of services given a training set. With regression, this neural network can then be judged on its accuracy with a hold-out data set.

The data that will be used is hosted by the University of California, Irvine [Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php?format=&task=&att=&area=&numAtt=&numIns=&type=text&sort=nameUp&view=table) and is called [Sentiment Labelled Sentences](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences). There are 3000 total reviews evenly distributed from Yelp, IMDb and Amazon, each already labelled with negative or positive sentiment. The benefit of this is that any number of the reviews can be used in a training set and can be used to show the benefit of small versus large training sets in neural networks.

I intend to use the TensorFlow API/library with Python 3.5 for my project.

# Expected Scope
With all of the data being labelled as negative or positive sentiment, the goal is to improve the classification system and develop a more advanced sentiment classification system. The goal is to apply sentiment labels of positive and negative as well as classifying the subject such as food/drink, movie, product or unmentioned. Since these two categories are independent of each other a proper way to determine accuracy of the system is required since a review can be correct, partially correct or incorrect and each should be weighted differently. This will also require a portion of the reviews to be labelled as a training set for the category classification. 

The biggest challenge I expect is the proper classification of a review subject. I suspect that many reviews will not mention a specific topic for their review and so the network will have to classify something based on the absence of information. This could prove more difficult than the positive and negative associated with sentiment since it is likely more present in a review.

The major milestones of this project are as follows:
1. Categorize the reviews categorically.
2. Isolate training and hold-out datasets.
3. Implement a text parser.
4. Implement neural network.
5. Properly process training dataset.
6. Run neural network against hold-out dataset.
7. Implement additional accuracy analysis methods if needed.

# Expected Outcomes
As mentioned before, I expected to be able to classify reviews sentiments as positive or negative as well as categorically with labels of food/drink, movie, product or unmentioned. With 3000 entries of data I expect to be able to use 40 or so entries as training data providing a great baseline for the neural network to learn and properly classify the data. I expect TensorFlow to have tools for analyzing the accuracy of the neural network but in the absence of those tools a regression test using the Mean Square Error method will be implemented to analyze the accuracy of the network against a hold-out data set.

# Key References
- N.A. (N.D.). TensorFlow. Retrieved from http://www.tensorflow.org
- Kassabgi, G. (Jan 16, 2017). Text Classification using Neural Networks. Retrieved from https://machinelearnings.co/text-classification-using-neural-networks-f5cd7b8765c6

