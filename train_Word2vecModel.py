# import modules & set up logging
import gensim
import os
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

##################### Training ########################

dotaCounter = 0
csCounter = 0
lolCounter = 0
othersCounter = 0

totalWordCounter = 0

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                # if counter < 5000:
                    #parse json msg
                try:
                    parsed = json.loads(line)
                    parsed_msg = parsed['msg']
                    parsed_game = parsed['game']
                except ValueError:
                    print "Error in line"
                    continue

                if parsed_game == "Dota%202":
                    global dotaCounter
                    dotaCounter += 1
                elif parsed_game == "League%20of%20Legends":
                    global lolCounter
                    lolCounter += 1
                elif parsed_game == "Counter-Strike%3A%20Global%20Offensive":
                    global csCounter
                    csCounter += 1
                else:
                    global othersCounter
                    othersCounter += 1


                # counter += 1
                # print counter
                parsed_msg = parsed_msg.split()

                global totalWordCounter
                totalWordCounter += len(parsed_msg)

                yield parsed_msg

                # else:
                #     break

# sentences = MySentences('txtfiles/messages')
sentences = MySentences('txtfiles/messages')

model = gensim.models.Word2Vec(sentences, workers=4)
model.save('models/02.01_default')

print dotaCounter, csCounter, lolCounter, othersCounter, totalWordCounter