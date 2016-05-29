from count_smilies_class import Emo
from collections import Counter
import os
import json
import codecs
import numpy as np

numOfTotalSentences = 0
messagecounter = 0
foundEmosAndSmilies = []

# emoCount = Emo(language="empty", emoticons=True, secondemo="emoticons")
emoCount = Emo(language="twitchstandard", emoticons=True, secondemo="emoticons")

# import messages and parse
def getData(filename="", dirname='txtfiles/messages'):
    messages = []
    if filename == "":
        for fname in os.listdir(dirname):
            print os.listdir(dirname), fname
            for line in open(os.path.join(dirname, fname)):
                try:
                    parsed = json.loads(line)
                    parsed_msg = parsed['msg']
                    messages.append(parsed_msg)

                except ValueError:
                    print "Error in line"
                    continue
    else:
        for line in open(os.path.join(dirname, filename)):
            try:
                parsed = json.loads(line)
                parsed_msg = parsed['msg']
                messages.append(parsed_msg)

            except ValueError:
                print "Error in line"
                continue

    return messages

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

messages, labels = openTwitterFile("txtfiles/labeledData/tweet_semevaltest.txt", "txtfiles/labeledData/tweet_semevaltest_so_score.txt")

# messages, labels = openOurFile("txtfiles/labeledData/TwitchDota.txt")
# print labeledMessages

# messages = getData('admiralbulldog.txt')
# print messages
# messages = getData('imaqtpie.txt')


"""
COUNT
"""
# for message in messages:
#     messagecounter += 1
#     # for every sentence
#     # score, words = emoCount.score(message) #score
#     words = emoCount.find_all(message) #count
#
#     foundEmosAndSmilies = foundEmosAndSmilies + words
#     if len(words) != 0:
#         numOfTotalSentences += 1
#         print(words)
#
#
# # in the end
# print("Messages total:")
# print messagecounter
# print("found smilies total:")
# print sum(Counter(foundEmosAndSmilies).values())
# print("found smilies examples ordered:")
# print Counter(foundEmosAndSmilies)
# print("found smilies top50")
# print Counter.most_common(Counter(foundEmosAndSmilies), 50)
# print("total number of sentences with smilies:")
# print numOfTotalSentences

"""
Classify
"""
for message in messages:
    messagecounter += 1
    # for every sentence
    score, words = emoCount.score(message)

    foundEmosAndSmilies = foundEmosAndSmilies + words
    if len(words) != 0:
        numOfTotalSentences += 1
        print(words)


# in the end
print("Messages total:")
print messagecounter
print("found smilies total:")
print sum(Counter(foundEmosAndSmilies).values())
print("found smilies examples ordered:")
print Counter(foundEmosAndSmilies)
print("found smilies top50")
print Counter.most_common(Counter(foundEmosAndSmilies), 50)
print("total number of sentences with smilies:")
print numOfTotalSentences