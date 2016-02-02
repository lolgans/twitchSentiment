# import modules & set up logging
from __future__ import division
import gensim
# import os
import json
import numpy as np
# from sklearn.manifold import TSNE
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from NNet.Nnet import NeuralNet
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# load Google's Word2vec model
# word2VecModel = gensim.models.Word2Vec.load_word2vec_format('/home/tobi/Downloads/GoogleNews-vectors-negative300.bin',
#                                                      binary=True)
# n_dim = 300

# or load individual model
# word2VecModel = gensim.models.Word2Vec.load('models/5k_MinCount2')
word2VecModel = gensim.models.Word2Vec.load('models/02.01_default')
n_dim = 100

# read file
with open("txtfiles/labeledData/labeledCsGoData_asArray.txt") as infile:
    labeledMessagesJson = infile.read()
    # load json Array
    messages = json.loads(labeledMessagesJson)

labeledMessages = []
labels = []

for message in messages:
    labeledMessages.append(message['msg'])

    # add labels. There are  2 possibilities: 1 add normal values from -5 to +5 (reduceLabelValues false), 2 reduce values to -1/0/1
    reduceLabelValues = True  # False
    if not reduceLabelValues:
        labels.append(label)
    elif reduceLabelValues:
        label = float(message['values'][0])  # convert to float, otherwise its not working
        if label < 0:
            labels.append(-1)
        elif label == 0:
            labels.append(0)
        elif label > 0:
            labels.append(1)
        else:
            print "Error converting labelValues"
            continue

# array must be floats/ other format
# print labels
labels = np.array(labels, dtype='float')
# print labels

# Binarize the output TODO
# from sklearn.preprocessing import label_binarize
# labels = label_binarize(labels, classes=[-1, 0, 1])
# n_classes = labels.shape[1]

# split into train and testSet
x_train, x_test, y_train, y_test = train_test_split(labeledMessages, labels, test_size=0.2)

# Do some very minor text preprocessing
def cleanText(corpus):
    corpus = [z.lower().replace('\n', '').split() for z in corpus]  # .lower()
    return corpus

x_train = cleanText(x_train)
x_test = cleanText(x_test)

# print x_train
# print y_train

# Build word vector for training set by using the average value of all word vectors in the message, then scale
def buildWordVector(text, size):
    vec = np.zeros(size).reshape((1, size))
    missing_words = []
    count = 0.
    for word in text:
        try:
            vec += word2VecModel[word].reshape((1, size))
            count += 1.
        except KeyError:
            missing_words.append(word)
            continue
    if count != 0:
        vec /= count
    # print missing_words
    # print len(missing_words)
    return vec

# Build train vectors then scale
train_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_train])
# print train_vecs
train_vecs = scale(train_vecs)
# print train_vecs

# Build test vectors then scale
test_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_test])
test_vecs = scale(test_vecs)

"""
Word Matching
"""
# from Afinn.afinn import Afinn
# afinn = Afinn(language="en", emoticons=True)
# rightSentimentCount = 0
# testWords = x_train + x_test  # Training- or TestSet or both
# testValues = np.concatenate((y_train, y_test))  # Training- or TestSetValues or both
# sentenceCount = len(testWords)
# for index, sent in enumerate(testWords):
#     score, missingWords = afinn.score(" ".join(sent))  # bisschen umgeschrieben, dass auch die missing words ausgegeben werden
#     if score > 0:
#         score = 1
#     elif score < 0:
#         score = -1
#
#     if testValues[index] == score:
#         rightSentimentCount += 1
#
# accuracy = rightSentimentCount / sentenceCount
# print 'Test Accuracy Word-Matching: %.2f' % accuracy


# Use classification algorithm (i.e. Stochastic Logistic Regression) on training set,
# then assess model performance on test set
from sklearn.multiclass import OneVsRestClassifier
# lr = OneVsRestClassifier(SGDClassifier(loss='log', penalty='l1'))
lr = SGDClassifier(loss='log', penalty='l1')
lr.fit(train_vecs, y_train)

print 'Test Accuracy SGD-Classifier: %.2f' % lr.score(test_vecs, y_test)




# # Create ROC curve
# pred_probas = lr.predict_proba(test_vecs)[:, 1]
#
# fpr, tpr, _ = roc_curve(y_test, pred_probas)
# roc_auc = auc(fpr, tpr)
#
# plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.legend(loc='lower right')
# plt.show()


nnet = NeuralNet(100, learn_rate=1e-1, penalty=1e-8)
maxiter = 1000
batch = 150
_ = nnet.fit(train_vecs, y_train, fine_tune=False, maxiter=maxiter, SGD=True, batch=batch, rho=0.9)

print 'Test Accuracy NNet: %.2f' % nnet.score(test_vecs, y_test)
