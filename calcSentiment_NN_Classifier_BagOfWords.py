# import modules & set up logging
from __future__ import division
import gensim
import os
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

# # load Google's Word2vec model
# word2VecModel = gensim.models.Word2Vec.load_word2vec_format('/home/tobi/Downloads/GoogleNews-vectors-negative300.bin',
#                                                      binary=True)
# n_dim = 300

# or load individual model
# word2VecModel = gensim.models.Word2Vec.load('models/5k_MinCount2')
# word2VecModel = gensim.models.Word2Vec.load('models/02.01_default')
# n_dim = 100

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


##################### Preprocess Data ########################

def getData(dirname):
    messages = []
    dotaCounter = 0
    lolCounter = 0
    csCounter = 0
    othersCounter = 0
    totalWordCounter = 0
    for fname in os.listdir(dirname):
        for line in open(os.path.join(dirname, fname)):
            try:
                parsed = json.loads(line)
                parsed_msg = parsed['msg']
                messages.append(parsed_msg)
                parsed_game = parsed['game']

                if parsed_game == "Dota%202":
                    dotaCounter += 1
                elif parsed_game == "League%20of%20Legends":
                    lolCounter += 1
                elif parsed_game == "Counter-Strike%3A%20Global%20Offensive":
                    csCounter += 1
                else:
                    othersCounter += 1

                # parsed_msg = parsed_msg.split() #  needed for Word2Vec

                totalWordCounter += len(parsed_msg)

            except ValueError:
                print "Error in line"
                continue

    return messages

# sentences = getData('txtfiles/messages')
# print sentences


# sentences = ['hallo was geht ab', 'what a fuck']



# # BagOfWords
# # from sklearn.feature_extraction.text import HashingVectorizer
# from sklearn.feature_extraction.text import CountVectorizer
#
# # Initialize the "CountVectorizer" object, which is scikit-learn's
# # bag of words tool.
# vectorizer = CountVectorizer(analyzer="word",
#                              tokenizer=None,
#                              preprocessor=None,
#                              stop_words=None)
#                              # max_features=5000)


# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.
# train_data_features = vectorizer.fit_transform(sentences)
# print train_data_features.shape

# Numpy arrays are easy to work with, so convert the result to an
# array
# train_data_features = train_data_features.toarray()
#
# train_data_features.shape


# Take a look at the words in the vocabulary
# vocab = vectorizer.get_feature_names()
# print vocab


# Sum up the counts of each vocabulary word
# dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it
# appears in the training set
# for tag, count in zip(vocab, dist):
#     print count, tag

# # Do some very minor text preprocessing
# def cleanText(corpus):
#     corpus = [z.lower().replace('\n', '').split() for z in corpus]  # .lower()
#     return corpus
#
# # Build word vector for training set by using the average value of all word vectors in the message, then scale
# def buildWordVector(text, size):
#     vec = np.zeros(size).reshape((1, size))
#     missing_words = []
#     count = 0.
#     for word in text:
#         try:
#             vec += word2VecModel[word].reshape((1, size))
#             count += 1.
#         except KeyError:
#             missing_words.append(word)
#             continue
#     if count != 0:
#         vec /= count
#     # print missing_words
#     # print len(missing_words)
#     return vec
#
# # Binarize the output TODO
# # from sklearn.preprocessing import label_binarize
# # labels = label_binarize(labels, classes=[-1, 0, 1])
# # n_classes = labels.shape[1]
#
# sgdAccuracies = []
# nnetAccuracies = []
#

# BagOfWords
# from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
vectorizer = CountVectorizer(analyzer="word",
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             ngram_range=(1, 2))
                             # max_features=5000)

# vectorizer = TfidfVectorizer(analyzer="word",
#                              tokenizer=None,
#                              preprocessor=None,
#                              stop_words=None)
                             # ngram_range=(1, 2))
                             # max_features=5000)

train_data_features = vectorizer.fit_transform(labeledMessages)
# print train_data_features.shape

# Numpy arrays are easy to work with, so convert the result to an
# array
train_data_features = train_data_features.toarray()

sgdAccuracies = []
nnetAccuracies= []

iterations = 100
for num in range(0, iterations):
    # split into train and testSet
    x_train, x_test, y_train, y_test = train_test_split(train_data_features, labels, test_size=0.2)

    # # Do some very minor text preprocessing
    # x_train = cleanText(x_train)
    # x_test = cleanText(x_test)

    # # Build train vectors then scale
    # train_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_train])
    # # print train_vecs
    # train_vecs = scale(train_vecs)
    # # print train_vecs

    # # Build test vectors then scale
    # test_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_test])
    # test_vecs = scale(test_vecs)

    # Use classification algorithm (i.e. Stochastic Logistic Regression) on training set,
    # then assess model performance on test set
    from sklearn.multiclass import OneVsRestClassifier
    # lr = OneVsRestClassifier(SGDClassifier(loss='log', penalty='l1'))
    lr = SGDClassifier(loss='log', penalty='l1')
    lr.fit(x_train, y_train)

    sgdScore = lr.score(x_test, y_test)
    sgdAccuracies.append(sgdScore)
    print 'Test Accuracy SGD-Classifier: %.2f' % sgdScore

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

    # NNet
    # nnet = NeuralNet(100, learn_rate=1e-1, penalty=1e-8)
    # maxiter = 1000
    # batch = 150
    # _ = nnet.fit(x_train, y_train, fine_tune=False, maxiter=maxiter, SGD=True, batch=batch, rho=0.9)
    #
    # nnetScore = nnet.score(x_test, y_test)
    # nnetAccuracies.append(nnetScore)
    # print 'Test Accuracy NNet: %.2f' % nnetScore
#
#
# """
# Word Matching
# """
# # from Afinn.afinn import Afinn
# # afinn = Afinn(language="en", emoticons=True)
# # rightSentimentCount = 0
# # testWords = x_train + x_test  # Training- or TestSet or both
# # testValues = np.concatenate((y_train, y_test))  # Training- or TestSetValues or both
# # sentenceCount = len(testWords)
# # for index, sent in enumerate(testWords):
# #     score, missingWords = afinn.score(" ".join(sent))  # bisschen umgeschrieben, dass auch die missing words ausgegeben werden
# #     if score > 0:
# #         score = 1
# #     elif score < 0:
# #         score = -1
# #
# #     if testValues[index] == score:
# #         rightSentimentCount += 1
# #
# # accuracy = rightSentimentCount / sentenceCount
# # print 'Test Accuracy Word-Matching: %.2f' % accuracy
#
print sgdAccuracies
print "SGD Average: %.2f" % np.mean(sgdAccuracies)
print "SGD Minimum: %.2f" % np.min(sgdAccuracies)
print "SGD Maximum: %.2f" % np.max(sgdAccuracies)
# print nnetAccuracies
# print "NNET Average: %.2f" % np.mean(nnetAccuracies)
# print "NNET Minimum: %.2f" % np.min(nnetAccuracies)
# print "NNET Maximum: %.2f" % np.max(nnetAccuracies)
