# This Python file uses the following encoding: utf-8

# import powerlaw
# import numpy as np
# data = np.array([3, 4, 5]) # data can be list or numpy array
# results = powerlaw.Fit(data)
# print results.power_law.alpha
# print results.power_law.xmin
# R, p = results.distribution_compare('power_law', 'lognormal')

from __future__ import division
from itertools import *
from pylab import *
from string import lower
from collections import Counter
import json
import os
import io


# The data: token counts from the Brown corpus
# tokens_with_count = Counter(imap(lower, brown.words()))
# counts = array(tokens_with_count.values())
# tokens = tokens_with_count.keys()

getMsgsBool = True
getUsersBool = False

# import messages and parse
def getMsgs(filename="", dirname='txtfiles/messages'):
    messages = []
    if filename == "":
        for fname in os.listdir(dirname):
            print os.listdir(dirname), fname
            for line in io.open(os.path.join(dirname, fname)):
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

def getUsers(filename="", dirname='txtfiles/messages'):
    users = []
    if filename == "":
        for fname in os.listdir(dirname):
            print os.listdir(dirname), fname
            for line in io.open(os.path.join(dirname, fname)):
                try:
                    parsed = json.loads(line)
                    parsed_user = parsed['user']
                    users.append(parsed_user)

                except ValueError:
                    print "Error in line"
                    continue
    else:
        for line in open(os.path.join(dirname, filename)):
            try:
                parsed = json.loads(line)
                parsed_user = parsed['user']
                users.append(parsed_user)

            except ValueError:
                print "Error in line"
                continue

    return users

if getMsgsBool:
    messages = getMsgs('imaqtpie.txt')

if getUsersBool:
    messages = getUsers('admiralbulldog.txt')

# bad variable names for users!!!
messagecounter = 0
kcount = 0
allwords = []

for message in messages:
    if kcount < 100:
        # print message
        messagecounter += 1
        # if getMsgsBool:
        newmessage = message.split(" ")
        # if getUsersBool:
        #     newmessage = message
        newallwords = allwords + newmessage
        allwords = newallwords
        #print allwords
        if (messagecounter%10000 == 0):
            kcount += 1
            print kcount * 10000
    else:
        break

tokens_with_count = Counter(allwords)
print tokens_with_count
print tokens_with_count.values()
print tokens_with_count.keys()
print len(tokens_with_count.values())
print len(tokens_with_count.keys())
counts = array(tokens_with_count.values())
print len(counts)
tokens = tokens_with_count.keys()

# tokens_with_count = Counter(imap(lower, brown.words()))
# counts = array(tokens_with_count.values())
# tokens = tokens_with_count.keys()

print messagecounter

# A Zipf plot
ranks = arange(1, len(counts)+1)
indices = argsort(-counts)
frequencies = counts[indices]
loglog(ranks, frequencies, marker=".")
title("rank/frequency plot")
xlabel("Rank")
ylabel("Word frequency")
grid(True)
for n in list(logspace(-0.5, log10(len(counts)), 20).astype(int)):
    print n
    # dummy = text(ranks[n], frequencies[n], " " + tokens[indices[n]],
    #              verticalalignment="bottom",
    #              horizontalalignment="left")

show()
