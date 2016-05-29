import gensim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# load Word2Vec model
model = gensim.models.Word2Vec.load('models/02.01_default')
# model = gensim.models.Word2Vec.load_word2vec_format('/home/tobi/Downloads/GoogleNews-vectors-negative300.bin', binary=True)


print model.doesnt_match("wtf fuck idiot hello".split())
print model.doesnt_match("4Head fuck idiot BibleThump".split())
print model.doesnt_match("wtf lol idiot omg".split())
print model.doesnt_match("gg funny rofl lol".split())
print model.doesnt_match("FailFish NotLikeThis BrokeBack EleGiggle".split()) #Brokeback und BibleThump hat er scheinbar falsch klassifiziert
print model.doesnt_match("BibleThump 4Head EleGiggle Brokeback".split())
print model.doesnt_match("Kappa KappaClaus KappaRoss KappaPride".split())

print model.similarity('hi', 'hello')
print model.similarity('omg', 'omfg')
print model.similarity('4Head', 'BibleThump')
print model.similarity('BrokeBack', 'BibleThump')
print model.similarity('Elegiggle', 'BrokeBack')
print model.similarity('Elegiggle', '4Head')
print model.similarity('EleGiggle', 'BibleThump')

print model.most_similar(positive=['wtf', 'lol'], negative=['omg'])
print model.most_similar(positive=['4Head', 'EleGiggle'], negative=['BibleThump']) #sehr seltsam zu EU??

# set vecs and word labels
# word_labels = ['4head', '2.is', '3.is', '4.is', '5.is']
# vecs = []
# vecs.append(model['4head'].reshape(1, 100))
# vecs.append(model['Anele'].reshape(1, 100))
# vecs.append(model['is'].reshape(1, 100))
# vecs.append(model['is'].reshape(1, 100))
# vecs.append(model['is'].reshape(1, 100))
# vecs = np.concatenate(vecs)
# vecs = np.array(vecs, dtype='float')
#
# # initalize TSNE
# ts = TSNE(2)
# reduced_vecs = ts.fit_transform(vecs)
#
# #  plot it
# fig = plt.figure()
# for i in range(len(reduced_vecs)):
#     color = 'g'
#     plt.plot(reduced_vecs[i, 0], reduced_vecs[i, 1], marker='o', color=color, markersize=8)
#     plt.annotate(
#         word_labels[i],
#         xy = (reduced_vecs[i, 0], reduced_vecs[i, 1]), xytext = (-20, 20),
#         textcoords = 'offset points', ha = 'right', va = 'bottom',
#         bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
#         arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
# plt.grid()
# plt.show()
