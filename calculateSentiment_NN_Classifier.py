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
from sklearn import cross_validation
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import codecs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

word2VecOur = True
word2VecGoogle = False
bow = False
bowBi = False
bowBoth = False
bowTri = False
bowTriBoth = False
tfidf = False


# # load Google's Word2vec model
if word2VecGoogle:
    word2VecModel = gensim.models.Word2Vec.load_word2vec_format('/home/tobi/Downloads/GoogleNews-vectors-negative300.bin',
                                                         binary=True)
    n_dim = 300

# or load individual model
if word2VecOur:
    word2VecModel = gensim.models.Word2Vec.load('models/02.01_default')
    n_dim = 100

vectorizer = None

if bow:
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 ngram_range=(1, 3),
                                 lowercase=False)

if tfidf:
    vectorizer = TfidfVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 ngram_range=(1, 2),
                                 lowercase=False
                                 )

if bowBi:
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 ngram_range=(2, 2),
                                 lowercase=False
                                 )

if bowBoth:
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 ngram_range=(1, 2),
                                 lowercase=False
                                 )

if bowTri:
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 ngram_range=(3, 3),
                                 lowercase=False
                                 )

if bowTriBoth:
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 ngram_range=(1, 3),
                                 lowercase=False
                                 )

"""
Read our Database
"""



def openOurFile(filename):
    labeledMessages = []
    labels = []
    with codecs.open(filename, encoding='UTF-8') as fid:
        for n, line in enumerate(fid):
            try:
                score, word = line.strip().split('\t')
            except ValueError:
                print 'Error in line %d of %s' % (n + 1, filename)

            labeledMessages.append(word)
            if score == 'NEUTRAL':
                score = 0
            if score == 'POSITIVE':
                score = 1
            if score == 'NEGATIVE':
                score = 2
            labels.append(score)

    labels = np.array(labels, dtype='float')
    return labeledMessages, labels

labeledMessages, labels = openOurFile("txtfiles/labeledData/TwitchDota.txt")

print labeledMessages
print labels

# read file old stuff
# with open("txtfiles/labeledData/labeledCsGoData_asArray.txt") as infile:
#     labeledMessagesJson = infile.read()
#     # load json Array
#     messages = json.loads(labeledMessagesJson)
#
# labeledMessages = []
# labels = []
#
# for message in messages:
#     labeledMessages.append(message['msg'])
#
#     # add labels. There are  2 possibilities: 1 add normal values from -5 to +5 (reduceLabelValues false), 2 reduce values to -1/0/1
#     reduceLabelValues = True  # False
#     if not reduceLabelValues:
#         labels.append(label)
#     elif reduceLabelValues:
#         label = float(message['values'][0])  # convert to float, otherwise its not working
#         if label < 0:
#             labels.append(-1)
#         elif label == 0:
#             labels.append(0)
#         elif label > 0:
#             labels.append(1)
#         else:
#             print "Error converting labelValues"
#             continue
#
# # array must be floats/ other format
# # print labels
# labels = np.array(labels, dtype='float')
# print labels

# Do some very minor text preprocessing
def cleanText(corpus):
    corpus = [z.lower().replace('\n', '').split() for z in corpus]  # .lower()
    return corpus

# Build word vector for training set by using the average value of all word vectors in the message, then scale
def buildWordVector(text, size):
    vec = np.zeros(size).reshape((1, size))
    missing_words = []
    count = 0.
    for word in text:
        try:
            vec += word2VecModel[word].reshape((1, size))
            # print vec, count
            count += 1.
        except KeyError:
            missing_words.append(word)
            continue
    if count != 0:
        vec /= count
    # print missing_words
    # print len(missing_words)
    return vec

# Binarize the output TODO
# from sklearn.preprocessing import label_binarize
# labels = label_binarize(labels, classes=[-1, 0, 1])
# n_classes = labels.shape[1]

sgdAccuracies = []
nnetAccuracies = []
svmAccuracies = []
linearSvmAccuracies = []
linearSvmAccuracies1 = []





# Machine Learning

# SVM (multiclass one vs one)
from sklearn import svm

svm1 = svm.SVC()

# SVM-Linear (multiclass one vs rest)
svmlin = svm.LinearSVC()
#
# NNet
nnet = NeuralNet(100, learn_rate=1e-1, penalty=1e-8)
maxiter = 1000
batch = 150

# SGD log regression
lr = SGDClassifier(loss='log', penalty='l1')

k_fold = cross_validation.KFold(len(labeledMessages), n_folds=10, shuffle=True)
for train_indices, test_indices in k_fold:

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for index in train_indices:
        x_train.append(labeledMessages[index])
        y_train.append(labels[index])

    for index in test_indices:
        x_test.append(labeledMessages[index])
        y_test.append(labels[index])

    # Do some very minor text preprocessing
    x_train = cleanText(x_train)
    x_test = cleanText(x_test)

    if word2VecGoogle | word2VecOur:
        print 'with Word2Vec'
        # Build train vectors then scale
        train_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_train])
        # print train_vecs.shape
        train_vecs2 = scale(train_vecs)
        # print train_vecs

        # Build test vectors then scale
        test_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_test])
        test_vecs2 = scale(test_vecs)

    if vectorizer is not None:
        print 'with Vectorizer'

        x_trainNew = []
        x_testNew = []

        for sentence in x_train:
            x_trainNew.append(" ".join(sentence))

        for sentence in x_test:
            x_testNew.append(" ".join(sentence))

        # print x_trainNew
        # x_trainNew = ['hallo das', 'hallo ist', 'hallo hallo test']

        train_vecs = vectorizer.fit_transform(x_trainNew)
        train_vecs2 = train_vecs.toarray()
        # train_vecs = scale(train_vecs)
        print train_vecs2.shape
        test_vecs = vectorizer.transform(x_testNew)
        test_vecs2 = test_vecs.toarray()
        # test_vecs = scale(test_vecs)
        print test_vecs.shape

    # labels as numpy array
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    print y_train.shape, y_test.shape

    # svm1.fit(train_vecs, y_train)
    # # print svm1.score(test_vecs, y_test)
    # svmAccuracies.append(svm1.score(test_vecs, y_test))

    svmlin.fit(train_vecs, y_train)
    # print svmlin.decision_function(test_vecs[0])
    # print svmlin.predict(test_vecs[0])
    linearSvmAccuracies.append(svmlin.score(test_vecs, y_test))

    nnet1 = NeuralNet(100, learn_rate=1e-1, penalty=1e-8)
    nnet1.fit(train_vecs2, y_train, fine_tune=False, maxiter=maxiter, SGD=True, batch=batch, rho=0.9)
    nnetScore = nnet1.score(test_vecs2, y_test)
    linearSvmAccuracies1.append(nnetScore)
    print 'Test Accuracy NNet: %.2f' % nnetScore
    # print(train_vecs.shape, y_train.shape)
    # print train_vecs2.shape
    # print y_train.shape
    nnet.fit(train_vecs2, y_train, fine_tune=False, maxiter=maxiter, SGD=True, batch=batch, rho=0.9)
    nnetScore = nnet.score(test_vecs2, y_test)
    nnetAccuracies.append(nnetScore)
    print 'Test Accuracy NNet: %.2f' % nnetScore

    # lr.fit(train_vecs, y_train)
    # sgdScore = lr.score(test_vecs, y_test)
    # sgdAccuracies.append(sgdScore)
    # print 'Test Accuracy SGD-Classifier: %.2f' % sgdScore

# cross_validation.cross_val_score(clf1, labeledMessages, labels, cv=kfold, n_jobs=-1)



# iterations = 100
# for num in range(0, iterations):
#     # split into train and testSet
#     x_train, x_test, y_train, y_test = train_test_split(labeledMessages, labels, test_size=0.2)
#
#     # Do some very minor text preprocessing
#     x_train = cleanText(x_train)
#     x_test = cleanText(x_test)
#
#     # Build train vectors then scale
#     train_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_train])
#     # print train_vecs
#     train_vecs = scale(train_vecs)
#     # print train_vecs
#
#     # Build test vectors then scale
#     test_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_test])
#     test_vecs = scale(test_vecs)
#
#     # Use classification algorithm (i.e. Stochastic Logistic Regression) on training set,
#     # then assess model performance on test set
#     # from sklearn.multiclass import OneVsRestClassifier
#     # lr = OneVsRestClassifier(SGDClassifier(loss='log', penalty='l1'))
#     lr = SGDClassifier(loss='log', penalty='l1')
#     lr.fit(train_vecs, y_train)
#
#     sgdScore = lr.score(test_vecs, y_test)
#     sgdAccuracies.append(sgdScore)
#     print 'Test Accuracy SGD-Classifier: %.2f' % sgdScore
#
#     # # Create ROC curve
#     # pred_probas = lr.predict_proba(test_vecs)[:, 1]
#     #
#     # fpr, tpr, _ = roc_curve(y_test, pred_probas)
#     # roc_auc = auc(fpr, tpr)
#     #
#     # plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
#     # plt.plot([0, 1], [0, 1], 'k--')
#     # plt.xlim([0.0, 1.0])
#     # plt.ylim([0.0, 1.05])
#     # plt.xlabel('False Positive Rate')
#     # plt.ylabel('True Positive Rate')
#     # plt.title('Receiver operating characteristic')
#     # plt.legend(loc='lower right')
#     # plt.show()
#
#     # SVM (multiclass one vs one)
#     from sklearn import svm
#     clf1 = svm.SVC()
#     clf1.fit(train_vecs, y_train)
#     # print clf.score(test_vecs, y_test)
#     svmAccuracies.append(clf1.score(test_vecs, y_test))
#
#
#     # SVM-Linear (multiclass one vs rest)
#     clf2 = svm.LinearSVC()
#     clf2.fit(train_vecs, y_train)
#     # print clf2.decision_function(test_vecs[0])
#     # print clf2.predict(test_vecs[0])
#     linearSvmAccuracies.append(clf2.score(test_vecs, y_test))
#
#
#     # SVM Light (Joachim)
#     # import svmlight
#     # # train a model based on the data
#     # model = svmlight.learn(x_train, type='classification', verbosity=0)
#     #
#     # # model data can be stored in the same format SVM-Light uses, for interoperability
#     # # with the binaries.
#     # svmlight.write_model(model, 'my_model.dat')
#     #
#     # # classify the test data. this function returns a list of numbers, which represent
#     # # the classifications.
#     # predictions = svmlight.classify(model, x_test)
#     # for p in predictions:
#     #     print '%.8f' % p
#
#     # NNet
#     # nnet = NeuralNet(100, learn_rate=1e-1, penalty=1e-8)
#     # maxiter = 1000
#     # batch = 150
#     # _ = nnet.fit(train_vecs, y_train, fine_tune=False, maxiter=maxiter, SGD=True, batch=batch, rho=0.9)
#     #
#     # nnetScore = nnet.score(test_vecs, y_test)
#     # nnetAccuracies.append(nnetScore)
#     # print 'Test Accuracy NNet: %.2f' % nnetScore


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

"""
Results
"""

# print sgdAccuracies
# print "SGD Average: %.2f" % np.mean(sgdAccuracies)
# print "SGD Minimum: %.2f" % np.min(sgdAccuracies)
# print "SGD Maximum: %.2f" % np.max(sgdAccuracies)
print nnetAccuracies
print "NNET Average: %.2f" % np.mean(nnetAccuracies)
print "NNET Minimum: %.2f" % np.min(nnetAccuracies)
print "NNET Maximum: %.2f" % np.max(nnetAccuracies)
# print svmAccuracies
# print "SVM Average: %.2f" % np.mean(svmAccuracies)
# print "SVM Minimum: %.2f" % np.min(svmAccuracies)
# print "SVM Maximum: %.2f" % np.max(svmAccuracies)
print linearSvmAccuracies
print "Linear-SVM Average: %.2f" % np.mean(linearSvmAccuracies)
print "Linear-SVM Minimum: %.2f" % np.min(linearSvmAccuracies)
print "Linear-SVM Maximum: %.2f" % np.max(linearSvmAccuracies)
print linearSvmAccuracies1
print "NNET1 Average: %.2f" % np.mean(linearSvmAccuracies)
print "NNET1 Minimum: %.2f" % np.min(linearSvmAccuracies)
print "NNET1 Maximum: %.2f" % np.max(linearSvmAccuracies)