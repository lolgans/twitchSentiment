# import modules & set up logging
import gensim
import os
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# load Google's Word2vec model
model = gensim.models.Word2Vec.load_word2vec_format('/home/tobi/Downloads/GoogleNews-vectors-negative300.bin', binary=True)

# load individual model
# model = gensim.models.Word2Vec.load('models/02.01_default')


with open('txtfiles/top500/cs_go_top500.txt', 'r') as infile:
    cs_go_words = infile.readlines()

with open('txtfiles/top500/dota2_top500.txt', 'r') as infile:
    dota2_words = infile.readlines()

with open('txtfiles/top500/lol_top500.txt', 'r') as infile:
    lol_words = infile.readlines()

with open('smilies/twitchstandard.txt', 'r') as infile:
    emos = infile.readlines()

def getWordVecs(allWords):
    vecs = []
    missing_words = []
    words_google_got = []
    word_labels = []
    for sentence in allWords:
        words = sentence.split(',')
        for idx, word in enumerate(words):
            word = word.replace('"', '')
            # word = word.replace('\n', '')
            # print word
            try:
                # if(word == '4head'):  # all same word vectors are the same
                #     print idx
                #     print model[word]
                #     print model[word].reshape(1, 100)

                vecs.append(model[word].reshape((1, 300)))
                # vecs.append(model[word].reshape((1, 100)))
                word_labels.append(word)
                words_google_got.append(word)
            except KeyError:
                missing_words.append(word)
                continue
    vecs = np.concatenate(vecs)
    # print words_google_got
    print missing_words
    print len(missing_words)
    return np.array(vecs, dtype='float'), word_labels  # TSNE expects float type values

# print cs_go_words

def getEmos(allWords):
    vecs = []
    missing_words = []
    words_google_got = []
    word_labels = []
    for sentence in allWords:
        words = sentence.split('\t')
        for idx, word in enumerate(words):
            if idx == 0:
                # word = word.replace('"', '')
                # word = word.replace('\n', '')
                # print word
                try:
                    # if(word == '4head'):  # all same word vectors are the same
                    #     print idx
                    #     print model[word]
                    #     print model[word].reshape(1, 100)

                    vecs.append(model[word].reshape((1, 300)))
                    # vecs.append(model[word].reshape((1, 100)))
                    word_labels.append(word)
                    words_google_got.append(word)
                except KeyError:
                    missing_words.append(word)
                    continue
    vecs = np.concatenate(vecs)
    # print words_google_got
    print missing_words
    return np.array(vecs, dtype='float'), word_labels  # TSNE expects float type values

cs_go_vecs, cs_go_word_labels = getWordVecs(cs_go_words)
dota2_vecs, dota2_word_labels = getWordVecs(dota2_words)
lol_vecs, lol_word_labels = getWordVecs(lol_words)
emo_vecs, emo_word_labels = getEmos(emos)

# print cs_go_vecs
# print cs_go_word_labels
# print lol_word_labels
# print dota2_vecs
# print lol_vecs

# print len(cs_go_vecs) + len(dota2_vecs) + len(lol_vecs)
# print len(np.concatenate((cs_go_vecs, dota2_vecs, lol_vecs)))
# print len(cs_go_word_labels + dota2_word_labels + lol_word_labels)

ts = TSNE(n_components=2, n_iter=10000)
reduced_vecs = ts.fit_transform(np.concatenate((cs_go_vecs, dota2_vecs, lol_vecs, emo_vecs)))
# reduced_vecs = ts.fit_transform(np.concatenate((cs_go_vecs, dota2_vecs, lol_vecs)))
word_labels = cs_go_word_labels + dota2_word_labels + lol_word_labels + emo_word_labels
# print len(word_labels)

fig = plt.figure()

# color points by word group to see if Word2Vec can separate them
for i in range(len(reduced_vecs)):
    # if i < len(cs_go_vecs):
    #     #cs_go words colored blue
    #     color = 'b'
    # elif len(cs_go_vecs) <= i < (len(dota2_vecs) + len(lol_vecs)):
    #     #dota2 words colored red
    #     color = 'r'
    # else:
    #     #lol words colored green
    #     color = 'g'
    if i < len(cs_go_vecs):
        #cs_go words colored blue
        color = 'b'
    elif len(cs_go_vecs) <= i < (len(dota2_vecs) + len(cs_go_vecs)):
        #dota2 words colored red
        color = 'r'
    elif (len(cs_go_vecs) + len(dota2_vecs)) <= i < (len(lol_vecs) + len(cs_go_vecs) + len(dota2_vecs)):
        #lol words colored green
        color = 'g'
    else:
        color = 'y'
    plt.plot(reduced_vecs[i, 0], reduced_vecs[i, 1], marker='o', color=color, markersize=8)
    plt.annotate(
        word_labels[i],
        xy = (reduced_vecs[i, 0], reduced_vecs[i, 1]), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = color, alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
plt.grid()
plt.show()
