import gensim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# load Word2Vec model
model = gensim.models.Word2Vec.load('models/5k_MinCount2')

word_labels = ['1.is', '2.is', '3.is', '4.is', '5.is']
vecs = []
vecs.append(model['is'].reshape(1, 100))
vecs.append(model['is'].reshape(1, 100))
vecs.append(model['is'].reshape(1, 100))
vecs.append(model['is'].reshape(1, 100))
vecs.append(model['is'].reshape(1, 100))

vecs = np.concatenate(vecs)
vecs = np.array(vecs, dtype='float')

ts = TSNE(2)
reduced_vecs = ts.fit_transform(vecs)

fig = plt.figure()

#  plot it
for i in range(len(reduced_vecs)):
    color = 'g'
    plt.plot(reduced_vecs[i, 0], reduced_vecs[i, 1], marker='o', color=color, markersize=8)
    plt.annotate(
        word_labels[i],
        xy = (reduced_vecs[i, 0], reduced_vecs[i, 1]), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
plt.grid()
plt.show()
