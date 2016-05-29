from count_smilies_class import Emo
from collections import Counter
import os
import json
import codecs
import numpy as np

numOfTotalSentences = 0
messagecounter = 0
foundEmosAndSmilies = []


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
                score = -1
            labels.append(score)

    labels = np.array(labels, dtype='float')
    return labeledMessages, labels

def openTwitterFile(filenameWords, filenameLabels):
    labeledMessages = []
    labels = []
    with codecs.open(filenameWords, encoding='UTF-8') as fid:
        for n, line in enumerate(fid):
            try:
                sentence = line.strip()
                # print sentence
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

# messages, labels = openTwitterFile("txtfiles/labeledData/tweet_semevaltest.txt", "txtfiles/labeledData/tweet_semevaltest_so_score.txt")

messages, labels = openOurFile("txtfiles/labeledData/TwitchDota.txt")

"""
Word Matching
"""
from Afinn.afinn import Afinn
afinn = Afinn(language="en", emoticons=False)
rightSentimentCount = 0
sentenceCount = 0
predictedlabels = []
emptysents = 0
print sentenceCount
print labels
# for index, sent in enumerate(messages):
#     sentenceCount += 1
#     print sentenceCount
#     # print sent
#     score, missingWords = afinn.score(sent)  # bisschen umgeschrieben, dass auch die missing words ausgegeben werden
#     a, b = afinn.find_all(sent)
#     if len(a) == 0:
#         emptysents += 1
#     # print score
#     if score > 0:
#         score = float(1.0)
#     elif score < 0:
#         score = float(-1.0)
#     print labels[index], score
#     if labels[index] == score:
#         rightSentimentCount += 1
#
#     predictedlabels.append(score)

"""
Smilie scoring
"""

from count_smilies_class import Emo
emoCount = Emo(language="twitchstandard", emoticons=False, secondemo="emoticons")
for index, sent in enumerate(messages):
    sentenceCount += 1
    # print sentenceCount
    print sent
    score, words = emoCount.score(sent)  # bisschen umgeschrieben, dass auch die missing words ausgegeben werden
    a = emoCount.find_all(sent)
    if len(a) == 0:
        emptysents += 1
    # print score
    if score > 0:
        score = float(1.0)
    elif score < 0:
        score = float(-1.0)
    print labels[index], score
    if labels[index] == score:
        rightSentimentCount += 1

    predictedlabels.append(score)



print rightSentimentCount
print sentenceCount
print float(rightSentimentCount) / sentenceCount
accuracy = float(rightSentimentCount) / sentenceCount
print 'Test Accuracy Word-Matching: %.2f' % accuracy

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# from sklearn.metrics import precision_recall_fscore_support

# print precision_recall_fscore_support(labels, predictedlabels, average='micro')
print precision_score(labels, predictedlabels, average='macro')
print recall_score(labels, predictedlabels, average='macro')
print f1_score(labels, predictedlabels, average='macro')

print(classification_report(labels, predictedlabels, target_names=['Negativ', 'Neutral', 'Positiv']))

print emptysents













