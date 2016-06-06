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

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

word2VecOur = False
word2VecGoogle = False
bow = False
bowBoth = False
tfidf = True

bowBi = False
bowTri = False
bowTriBoth = False



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
                                 ngram_range=(1, 1),
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

def openTwitterFile(filenameWords, filenameLabels):
    labeledMessages = []
    labels = []
    with codecs.open(filenameWords, encoding='UTF-8') as fid:
        for n, line in enumerate(fid):
            try:
                sentence = line.strip()
                print sentence
            except ValueError:
                print 'Error in line %d of %s' % (n + 1, filenameWords)

            labeledMessages.append(sentence)
            # if score == 'NEUTRAL':
            #     score = 0
            # if score == 'POSITIVE':
            #     score = 1
            # if score == 'NEGATIVE':
            #     score = -1
            # labels.append(score)

    with codecs.open(filenameLabels, encoding='UTF-8') as fid:
        for n, line in enumerate(fid):
            try:
                label = line
            except ValueError:
                print 'Error in line %d of %s' % (n + 1, filenameWords)

            labels.append(label)

    labels = np.array(labels, dtype='float')
    # print labeledMessages
    return labeledMessages, labels

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
                score = 1
            if score == 'POSITIVE':
                score = 2
            if score == 'NEGATIVE':
                score = 0
            labels.append(score)

    labels = np.array(labels, dtype='float')
    return labeledMessages, labels

# labeledMessages, labels = openOurFile("txtfiles/labeledData/TwitchDota.txt")
labeledMessages, labels = openTwitterFile("txtfiles/labeledData/tweet_semevaltest.txt", "txtfiles/labeledData/tweet_semevaltest_so_score.txt")


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


nnetAccuracies = []
nnetPrecision = []
nnetRecall = []
nnetF1 = []
nnetPrecisionClasses = []
nnetRecallClasses = []
nnetF1Classes = []


svmAccuracies = []
svmPrecision = []
svmRecall = []
svmF1 = []
svmPrecisionClasses = []
svmRecallClasses = []
svmF1Classes = []

linearSvmAccuracies = []
linearSvmPrecision = []
linearSvmRecall = []
linearSvmF1 = []
linearSvmPrecisionClasses = []
linearSvmRecallClasses = []
linearSvmF1Classes = []





# Machine Learning

# SVM (multiclass one vs one)
from sklearn import svm

svm1 = svm.SVC(kernel = 'rbf', C = 1000, gamma = 0.001)

# SVM-Linear (multiclass one vs rest)
svmlin = svm.LinearSVC(C=1)
#
# NNet
nnet = NeuralNet(100, learn_rate=1e-1, penalty=1e-8)
maxiter = 1000
batch = 150

# SGD log regression
lr = SGDClassifier(loss='log', penalty='l1')

k_fold = cross_validation.KFold(len(labeledMessages), n_folds=2, shuffle=True)
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
    # print y_train.shape, y_test.shape


    """
    Test
    """

    # from sklearn.cross_validation import train_test_split
    # from sklearn.grid_search import GridSearchCV
    # from sklearn.metrics import classification_report
    # from sklearn.svm import SVC
    # # from __future__ import print_function
    #
    # print(__doc__)
    # # Split the dataset in two equal parts
    # # X_train, X_test, y_train, y_test = train_test_split(
    # #     X, y, test_size=0.5, random_state=0)
    #
    # # Set the parameters by cross-validation
    # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
    #                      'C': [1, 10, 100, 1000]},
    #                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
    #                     {'kernel': ['poly'], 'C': [1, 10, 100, 1000]}]
    #
    # # scores = ['precision', 'recall']
    # scores = ['accuracy', 'f1_macro']
    #
    # for score in scores:
    #     print("# Tuning hyper-parameters for %s" % score)
    #     print()
    #
    #     clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
    #                        #scoring='%s_weighted' % score)
    #                        scoring=score)
    #     clf.fit(train_vecs, y_train)
    #
    #     print("Best parameters set found on development set:")
    #     print()
    #     print(clf.best_params_)
    #     print()
    #     print("Grid scores on development set:")
    #     print()
    #     for params, mean_score, scores in clf.grid_scores_:
    #         print("%0.3f (+/-%0.03f) for %r"
    #               % (mean_score, scores.std() * 2, params))
    #     print()
    #
    #     print("Detailed classification report:")
    #     print()
    #     print("The model is trained on the full development set.")
    #     print("The scores are computed on the full evaluation set.")
    #     print()
    #     y_true, y_pred = y_test, clf.predict(test_vecs)
    #     print(classification_report(y_true, y_pred))
    #     print()


    svm1.fit(train_vecs, y_train)
    # print svm1.score(test_vecs, y_test)
    # print test_vecs
    svmpredictions = svm1.predict(test_vecs)
    # svm1prediction = svm1prediction
    svmAccuracies.append(svm1.score(test_vecs, y_test))
    svmPrecision.append(precision_score(y_test, svmpredictions, average='macro'))
    svmRecall.append(recall_score(y_test, svmpredictions, average='macro'))
    svmF1.append(f1_score(y_test, svmpredictions, average='macro'))
    svmPrecisionClasses.append(precision_score(y_test, svmpredictions, average=None))
    svmRecallClasses.append(recall_score(y_test, svmpredictions, average=None))
    svmF1Classes.append(f1_score(y_test, svmpredictions, average=None))

    # print precision_score(y_test, svm1prediction, average='macro')
    # print recall_score(y_test, svm1prediction, average='macro')
    # print f1_score(y_test, svm1prediction, average='macro')
    #
    # print(classification_report(y_test, svm1prediction, target_names=['Negativ', 'Neutral', 'Positiv']))

    svmlin.fit(train_vecs, y_train)
    # print svmlin.decision_function(test_vecs[0])
    # print svmlin.predict(test_vecs[0])
    svmlinpredictions = svmlin.predict(test_vecs)

    linearSvmAccuracies.append(svmlin.score(test_vecs, y_test))
    linearSvmPrecision.append(precision_score(y_test, svmlinpredictions, average='macro'))
    linearSvmRecall.append(recall_score(y_test, svmlinpredictions, average='macro'))
    linearSvmF1.append(f1_score(y_test, svmlinpredictions, average='macro'))
    linearSvmPrecisionClasses.append(precision_score(y_test, svmlinpredictions, average=None))
    linearSvmRecallClasses.append(recall_score(y_test, svmlinpredictions, average=None))
    linearSvmF1Classes.append(f1_score(y_test, svmlinpredictions, average=None))

    # print precision_score(y_test, svmlinprediction, average='macro')
    # print recall_score(y_test, svmlinprediction, average='macro')
    # print f1_score(y_test, svmlinprediction, average='macro')
    # print f1_score(y_test, svmlinprediction, average=None)
    #
    # print(classification_report(y_test, svmlinpredictions, target_names=['Negativ', 'Neutral', 'Positiv']))
    #
    # print x_test[0], x_test[1]
    # nnet1 = NeuralNet(100, learn_rate=1e-1, penalty=1e-8)
    # nnet1.fit(train_vecs2, y_train, fine_tune=False, maxiter=maxiter, SGD=True, batch=batch, rho=0.9)
    # nnetScore, nnetpredictions = nnet1.score(test_vecs2, y_test)
    #
    # nnetAccuracies.append(nnetScore)
    # nnetPrecision.append(precision_score(y_test, nnetpredictions, average='macro'))
    # nnetRecall.append(recall_score(y_test, nnetpredictions, average='macro'))
    # nnetF1.append(f1_score(y_test, nnetpredictions, average='macro'))
    # nnetPrecisionClasses.append(precision_score(y_test, nnetpredictions, average=None))
    # nnetRecallClasses.append(recall_score(y_test, nnetpredictions, average=None))
    # nnetF1Classes.append(f1_score(y_test, nnetpredictions, average=None))




    # overfits this way
    # nnet.fit(train_vecs2, y_train, fine_tune=False, maxiter=maxiter, SGD=True, batch=batch, rho=0.9)
    # nnetScore = nnet.score(test_vecs2, y_test)
    # nnetAccuracies.append(nnetScore)
    # print 'Test Accuracy NNet: %.2f' % nnetScore

    # lr.fit(train_vecs, y_train)
    # sgdScore = lr.score(test_vecs, y_test)
    # sgdAccuracies.append(sgdScore)
    # print 'Test Accuracy SGD-Classifier: %.2f' % sgdScore

# cross_validation.cross_val_score(clf1, labeledMessages, labels, cv=kfold, n_jobs=-1)

"""
Results
"""

# """ NNET"""
# print nnetAccuracies
# print "NNET Average: %.2f" % np.mean(nnetAccuracies)
# print "NNET Average-Minimum: %.2f" % np.min(nnetAccuracies)
# print "NNET Average- Maximum: %.2f" % np.max(nnetAccuracies)
# print "NNET Precision: %.2f" % np.mean(nnetPrecision)
# print "NNET Recall: %.2f" % np.mean(nnetRecall)
# print "NNET F1: %.2f" % np.mean(nnetF1)
#
# #calc class precision
# cl1 = []
# cl2 = []
# cl3 = []
# for metric in nnetPrecisionClasses:
#     cl1.append(metric[0])
#     cl2.append(metric[1])
#     cl3.append(metric[2])
#
# print "NNET class1 Precision: %.2f" % np.mean(cl1)
# print "NNET class2 Precision: %.2f" % np.mean(cl2)
# print "NNET class3 Precision: %.2f" % np.mean(cl3)
# print np.mean(cl1 +cl2 +cl3)
#
# #calc class recall
# cl1 = []
# cl2 = []
# cl3 = []
# for metric in nnetRecallClasses:
#     cl1.append(metric[0])
#     cl2.append(metric[1])
#     cl3.append(metric[2])
#
# print "NNET class1 Recall: %.2f" % np.mean(cl1)
# print "NNET class2 Recall: %.2f" % np.mean(cl2)
# print "NNET class3 Recall: %.2f" % np.mean(cl3)
# print np.mean(cl1 +cl2 +cl3)
#
# #calc class f1
# cl1 = []
# cl2 = []
# cl3 = []
# for metric in nnetF1Classes:
#     cl1.append(metric[0])
#     cl2.append(metric[1])
#     cl3.append(metric[2])
#
# print "NNET class1 F1: %.2f" % np.mean(cl1)
# print "NNET class2 F2: %.2f" % np.mean(cl2)
# print "NNET class3 F3: %.2f" % np.mean(cl3)
#
# print np.mean(cl1 +cl2 +cl3)

"""SVM RBF"""

print svmAccuracies
print "svm Average: %.2f" % np.mean(svmAccuracies)
print "svm Average-Minimum: %.2f" % np.min(svmAccuracies)
print "svm Average- Maximum: %.2f" % np.max(svmAccuracies)
print "svm Precision: %.2f" % np.mean(svmPrecision)
print "svm Recall: %.2f" % np.mean(svmRecall)
print "svm F1: %.2f" % np.mean(svmF1)

#calc class precision
pcl1 = []
pcl2 = []
pcl3 = []
for metric in svmPrecisionClasses:
    pcl1.append(metric[0])
    pcl2.append(metric[1])
    pcl3.append(metric[2])

print "svm class1 Precision: %.2f" % np.mean(pcl1)
print "svm class2 Precision: %.2f" % np.mean(pcl2)
print "svm class3 Precision: %.2f" % np.mean(pcl3)
print np.mean(pcl1 +pcl2 +pcl3)

#calc class recall
rcl1 = []
rcl2 = []
rcl3 = []
for metric in svmRecallClasses:
    rcl1.append(metric[0])
    rcl2.append(metric[1])
    rcl3.append(metric[2])

print "svm class1 Recall: %.2f" % np.mean(rcl1)
print "svm class2 Recall: %.2f" % np.mean(rcl2)
print "svm class3 Recall: %.2f" % np.mean(rcl3)
print np.mean(rcl1 +rcl2 +rcl3)

#calc class f1
cl1 = []
cl2 = []
cl3 = []
for metric in svmF1Classes:
    cl1.append(metric[0])
    cl2.append(metric[1])
    cl3.append(metric[2])

print "svm class1 F1: %.2f" % np.mean(cl1)
print "svm class2 F2: %.2f" % np.mean(cl2)
print "svm class3 F3: %.2f" % np.mean(cl3)

print np.mean(cl1 +cl2 +cl3)

"""LINEAR SVM"""
print linearSvmAccuracies
print "linearSvm Average: %.2f" % np.mean(linearSvmAccuracies)
print "linearSvm Average-Minimum: %.2f" % np.min(linearSvmAccuracies)
print "linearSvm Average- Maximum: %.2f" % np.max(linearSvmAccuracies)
print "linearSvm Precision: %.2f" % np.mean(linearSvmPrecision)
print "linearSvm Recall: %.2f" % np.mean(linearSvmRecall)
print "linearSvm F1: %.2f" % np.mean(linearSvmF1)

#calc class precision
cl1 = []
cl2 = []
cl3 = []
for metric in linearSvmPrecisionClasses:
    cl1.append(metric[0])
    cl2.append(metric[1])
    cl3.append(metric[2])

print "linearSvm class1 Precision: %.2f" % np.mean(cl1)
print "linearSvm class2 Precision: %.2f" % np.mean(cl2)
print "linearSvm class3 Precision: %.2f" % np.mean(cl3)
print np.mean(cl1 +cl2 +cl3)

#calc class recall
cl1 = []
cl2 = []
cl3 = []
for metric in linearSvmRecallClasses:
    cl1.append(metric[0])
    cl2.append(metric[1])
    cl3.append(metric[2])

print "linearSvm class1 Recall: %.2f" % np.mean(cl1)
print "linearSvm class2 Recall: %.2f" % np.mean(cl2)
print "linearSvm class3 Recall: %.2f" % np.mean(cl3)
print np.mean(cl1 +cl2 +cl3)

#calc class f1
cl1 = []
cl2 = []
cl3 = []
for metric in linearSvmF1Classes:
    cl1.append(metric[0])
    cl2.append(metric[1])
    cl3.append(metric[2])

print "linearSvm class1 F1: %.2f" % np.mean(cl1)
print "linearSvm class2 F2: %.2f" % np.mean(cl2)
print "linearSvm class3 F3: %.2f" % np.mean(cl3)

print np.mean(cl1 +cl2 +cl3)


print "AcHTUNG AUF KLASSEN ACHTEN, ob -1 0 und 1 oder 0 1 2"







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

