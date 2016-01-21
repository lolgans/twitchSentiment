# import modules & set up logging
import gensim
import os
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

################### Standard Vector von Google ########################

# model = gensim.models.Word2Vec.load_word2vec_format('/home/tobi/Downloads/GoogleNews-vectors-negative300.bin', binary=True)
# print model['gg']
# print model['wtf']
# print model['fu']
# print model['fuck']
# print model['fucking']
# print model['vac']
# print model['rip']
# print model['recast']
# print model['ebola']
# print model['titan']


##################### Training ########################

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            counter = 0
            for line in open(os.path.join(self.dirname, fname)):
                if counter < 5000:
                    #parse json msg
                    parsed_msg = json.loads(line)['msg']
                    print parsed_msg.split()
                    counter += 1
                    print counter
                    yield parsed_msg.split()
                else:
                    break

sentences = MySentences('txtfiles/messages')
print list(sentences)

model = gensim.models.Word2Vec(sentences, min_count=2, workers=4)
model.save('models/5k_MinCount2')

#################### Visualize ########################

# model = gensim.models.Word2Vec.load('models/50k_default')
#
# word_vectors = []
# word_labels = []
# for word in model.vocab:
#     word_vectors.append(model[word])
#     # print word
#     word_labels.append(word)
#
# print len(word_vectors)
# print len(word_labels)
#
# vectors = np.asfarray(word_vectors, dtype='float')
#
# tsne = TSNE(2)
# reduced_vecs = tsne.fit_transform(vectors)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# # print reduced_vecs
# for i in range(len(reduced_vecs)):
#     print word_labels[i], reduced_vecs[i]
#     plt.plot(reduced_vecs[i, 0], reduced_vecs[i, 1], marker='o', color='g', markersize=8)
#     # ax.annotate(word_labels[i], xy=(reduced_vecs[i, 0], reduced_vecs[i, 1]), textcoords='offset points')
# plt.show()




# LIFE [ -1.84802806  12.13029876]
# LATER [ 0.62566132  4.63435847]
# all [ 1.41492249 -9.06423481]
# DELAY [ -3.0725052  -17.61097958]
# aegis [  7.80232218 -23.93532369]
# asian [ 20.09263282 -12.17862226]