# import modules & set up logging
import gensim
import os
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = gensim.models.Word2Vec.load_word2vec_format('/home/tobi/Downloads/GoogleNews-vectors-negative300.bin', binary=True)


with open('txtfiles/cs_go_top500.txt', 'r') as infile:
    cs_go_words = infile.readlines()

with open('txtfiles/dota2_top500.txt', 'r') as infile:
    dota2_words = infile.readlines()

with open('txtfiles/lol_top500.txt', 'r') as infile:
    lol_words = infile.readlines()

def getWordVecs(allWords):
    vecs = []
    # counter = 0
    # missing_words = []
    # words_google_got = []
    word_labels = []
    for sentence in allWords:
        words = sentence.split(',')
        for word in words:
            word = word.replace('"', '')
            # word = word.replace('\n', '')
            # print word
            try:
                vecs.append(model[word].reshape((1, 300)))
                word_labels.append(word)
                # words_google_got.append(word)
                # counter += 1
            except KeyError:
                # missing_words.append(word)
                continue
    vecs = np.concatenate(vecs)
    return np.array(vecs, dtype='float'), word_labels #TSNE expects float type values

# print cs_go_words

cs_go_vecs, cs_go_word_labels = getWordVecs(cs_go_words)
dota2_vecs, dota2_word_labels = getWordVecs(dota2_words)
lol_vecs, lol_word_labels = getWordVecs(lol_words)

# print cs_go_vecs
print cs_go_word_labels
print lol_word_labels
# print dota2_vecs
# print lol_vecs

ts = TSNE(2)
reduced_vecs = ts.fit_transform(np.concatenate((cs_go_vecs, dota2_vecs, lol_vecs)))
word_labels = cs_go_word_labels + dota2_word_labels + lol_word_labels
print len(word_labels)

fig = plt.figure()
ax = fig.add_subplot(111)

#color points by word group to see if Word2Vec can separate them
for i in range(len(reduced_vecs)):
    if i < len(cs_go_vecs):
        #cs_go words colored blue
        color = 'b'
    elif i >= len(cs_go_vecs) and i < (len(dota2_vecs) + len(lol_vecs)):
        #dota2 words colored red
        color = 'r'
    else:
        #lol words colored green
        color = 'g'
    plt.plot(reduced_vecs[i, 0], reduced_vecs[i, 1], marker='o', color=color, markersize=8)
    ax.annotate(word_labels[i], xy=(reduced_vecs[i, 0], reduced_vecs[i, 1]), textcoords='offset points')
plt.show()
