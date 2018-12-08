import glob
import numpy as np
import os

oneb = np.zeros(50)
oneb[0] = 1
twob = np.zeros(50)
twob[12] = 1
threeb = np.zeros(50)
threeb[25] = 1
fourb = np.zeros(50)
fourb[37] = 1
anchorwords = {'the':oneb, 'and':twob, 'are':threeb, 'is':fourb}


def loadfromfile(filename):
    fileembeds = {}
    with open(filename) as file:
        sepd = file.read().split(']')
        for line in sepd[:-1]:
            line = line.replace('[','')
            comps = line.split()
            embvec = [float(f) for f in comps[1:]]
            fileembeds[comps[0]] = embvec
    return fileembeds

def transform(embed, words, targets):
    anmat = []
    preanmat = []
    for word in words.keys():
        anmat.append(embed[word])
        preanmat.append(words[word])
    anmat = np.array(anmat)
    A, res, rank, s = np.linalg.lstsq(anmat, preanmat)
    allmat = []
    allkey = []
    for w,e in embed.items():
        allmat.append([float(i) for i in e])
        allkey.append(w)
    trans2 = np.dot(np.array(list(allmat)), A)
    transembeds = {}
    for i in range(len(allkey)):
        transembeds[allkey[i]] = trans2[i]
    return transembeds

def makedicts():
    trainroot = "./article_embeds/"
    testroot = "./article_test_embeds/"
    trains = glob.glob(trainroot + '*.txt')
    tests = glob.glob(testroot + '*.txt')
    traind = {}
    testd = {}
    first = True
    targets = {}
    for t in trains:
        embeds = loadfromfile(t)
        if first:
            traind[os.path.basename(t)] = embeds
            for word in anchorwords:
                targets[word] = embeds[word]
        else:
            traind[os.path.basename(t)] = transform(embeds, anchorwords, targets)
    for t in tests:
        embeds = loadfromfile(t)
        testd[os.path.basename(t)] = transform(embeds, anchorwords, targets)
    return traind, testd


def cumdif(traind, testd):
    cumdifs = {}
    cumcounts = {}
    trainsets = traind.keys()
    for fname, embeds in testd.items():
        filecumdifs = {}
        filecumcounts = {}
        for f in trainsets:
            filecumdifs[f] = 0.
            filecumcounts[f] = 0.
        for word, vec in embeds.items():
            for tset in trainsets:
                newvec = traind[tset].get(word)
                if newvec is not None:
                    filecumdifs[tset] += np.linalg.norm(np.array(vec) - np.array(newvec))
                    filecumcounts[tset] += 1.
        for w,v in filecumdifs.items():
            filecumdifs[w] = v / filecumcounts[w]
        cumdifs[fname] = filecumdifs
    return cumdifs

traind, testd = makedicts()
cumdifs = cumdif(traind, testd)

for fname, difs in cumdifs.items():
    print(fname + ":")
    sd = sorted(difs, key=lambda k: difs[k])
    for tname in sd[:3]:
        print("Cumulative difference " + tname + ":" + str(difs[tname]))
