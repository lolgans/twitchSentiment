# import modules & set up logging
import gensim
import os
import json
from sklearn.manifold import TSNE
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

pos_tweets = [['hallo', 'du'], ['nein', 'idiot'], ['ja']]
neg_tweets = [['das', 'ist'], ['die', 'mareike'], ['doof']]

#use 1 for positive sentiment, 0 for negative
y = np.concatenate((np.ones(len(pos_tweets)), np.zeros(len(neg_tweets))))
print y

x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_tweets, neg_tweets)), y, test_size=0.2)
print x_train
print x_test
print y_train
print y_test