import zipfile
import collections
import numpy as np

import math
import random
import collections
import os
import string
from sklearn.decomposition import PCA

import glob

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as Func
from torch.optim.lr_scheduler import StepLR
import time


data_index = 0
class Options(object):
  def __init__(self, datafile, vocabulary_size):
    self.vocabulary_size = vocabulary_size
    self.vocabulary = self.read_data(datafile)
    data_or, self.count, self.vocab_words = self.build_dataset(self.vocabulary,
                                                              self.vocabulary_size)
    self.train_data = self.subsampling(data_or)
    self.sample_table = self.init_sample_table()

  def read_data(self,filename):
    translator = str.maketrans("", "", string.punctuation)
    with open(filename) as f:
      data = f.read().translate(translator).split()
      data = [x.lower() for x in data if x != 'eoood'][:5000]
    return data

  def build_dataset(self,words, n_words):
    """Process raw inputs into a ."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
      dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
      if word in dictionary:
        index = dictionary[word]
      else:
        index = 0  # dictionary['UNK']
        unk_count += 1
      data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, reversed_dictionary

  def init_sample_table(self):
    count = [ele[1] for ele in self.count]
    pow_frequency = np.array(count)**0.75
    power = sum(pow_frequency)
    ratio = pow_frequency/ power
    table_size = 1e8
    count = np.round(ratio*table_size)
    sample_table = []
    for idx, x in enumerate(count):
      sample_table += [idx]*int(x)
    return np.array(sample_table)
  def weight_table(self):
    count = [ele[1] for ele in self.count]
    pow_frequency = np.array(count)**0.75
    power = sum(pow_frequency)
    ratio = pow_frequency/ power
    return np.array(ratio)
  def subsampling(self,data):
    count = [ele[1] for ele in self.count]
    frequency = np.array(count)/sum(count)
    P = dict()
    for idx, x in enumerate(frequency):
      if x != 0:
          y = (math.sqrt(x/0.001)+1)*0.001/x
      else:
          y = (math.sqrt(0.000000000000001/0.001)+1)*0.001/0.0000000000000001
      P[idx] = y
    subsampled_data = list()
    for word in data:
      if random.random()<P[word]:
        subsampled_data.append(word)
    return subsampled_data



  def generate_batch2(self, skip_window, batch_size):
    global data_index
    data = self.train_data
    batch = np.ndarray(shape=(batch_size), dtype=np.int64)
    labels = np.ndarray(shape=(batch_size, 2 * skip_window), dtype=np.int64)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)

    if data_index + span > len(data):
      data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size):
      batch[i] = buffer[skip_window]
      targets = [x for x in range(skip_window)] + [x for x in range(skip_window + 1, span)]
      for idj, j in enumerate(targets):
        labels[i,idj] = buffer[j]
      if data_index == len(data):
        buffer.extend(data[:span])
        data_index = span
        self.process = False
      else:
        buffer.append(data[data_index])
        data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

  def generate_batch(self, window_size, batch_size, count):
    data = self.train_data
    global data_index
    span = 2 * window_size + 1
    context = np.ndarray(shape=(batch_size, 2 * window_size), dtype=np.int64)
    labels = np.ndarray(shape=(batch_size), dtype=np.int64)
    pos_pair = []

    if data_index + span > len(data):
      data_index = 0
      self.process = False
    buffer = data[data_index:data_index + span]
    pos_u = []
    pos_v = []

    for i in range(batch_size):
      data_index += 1
      context[i,:] = buffer[:window_size] + buffer[window_size + 1:]
      labels[i] = buffer[window_size]
      if data_index + span > len(data):
        buffer[:] = data[:span]
        data_index = 0
        self.process = False
      else:
        buffer = data[data_index:data_index + span]

      for j in range(span-1):
        pos_u.append(labels[i])
        pos_v.append(context[i,j])
    neg_v = np.random.choice(self.sample_table, size=(batch_size * 2 * window_size, count))
    return np.array(pos_u), np.array(pos_v), neg_v




import json, csv
from scipy.stats import spearmanr
import math
def cosine_similarity(v1,v2):
  "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
  sumxx, sumxy, sumyy = 0, 0, 0
  for i in range(len(v1)):
    x = v1[i]; y = v2[i]
    sumxx += x * x
    sumyy += y * y
    sumxy += x * y
  return sumxy/math.sqrt(sumxx*sumyy)

def scorefunction(embed):
  f = open('./vocab.txt')
  line = f.readline()
  vocab = []
  wordindex = dict()
  index = 0
  while line:
    word = line.strip().split()[0]
    wordindex[word] = index
    index = index + 1
    line = f.readline()
  f.close()
  ze = []
  with open('./combined.csv') as csvfile:
    filein = csv.reader(csvfile)
    index = 0
    consim = []
    humansim = []
    for eles in filein:
      if index == 0:
        index = 1
        continue
      if (eles[0] not in wordindex) or (eles[1] not in wordindex):
        continue

      word1 = int(wordindex[eles[0]])
      word2 = int(wordindex[eles[1]])
      humansim.append(float(eles[2]))


      value1 =  embed[word1]
      value2 =  embed[word2]
      index =index + 1
      score = cosine_similarity(value1, value2)
      consim.append(score)


  cor1, pvalue1 = spearmanr(humansim, consim)


  return cor1

class skipgram(nn.Module):
  def __init__(self, vocab_size, embedding_dim):
    super(skipgram, self).__init__()
    self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
    self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
    self.embedding_dim = embedding_dim
    self.init_emb()
  def init_emb(self):
    initrange = 0.5 / self.embedding_dim
    self.u_embeddings.weight.data.uniform_(-initrange, initrange)
    self.v_embeddings.weight.data.uniform_(-0, 0)
  def forward(self, u_pos, v_pos, v_neg, batch_size):

    embed_u = self.u_embeddings(u_pos)
    embed_v = self.v_embeddings(v_pos)

    score  = torch.mul(embed_u, embed_v)
    score = torch.sum(score, dim=1)
    log_target = Func.logsigmoid(score).squeeze()

    neg_embed_v = self.v_embeddings(v_neg)

    neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()
    neg_score = torch.sum(neg_score, dim=1)
    sum_log_sampled = Func.logsigmoid(-1*neg_score).squeeze()

    loss = log_target + sum_log_sampled

    return -1 * loss.sum()/batch_size
  def input_embeddings(self):
    return self.u_embeddings.weight.data.cpu().numpy()
  def save_embedding(self, file_name, id2word):
    embeds = self.u_embeddings.weight.data.cpu().numpy()
    fo = open(file_name, 'w')
    for idx in range(len(embeds)):
      word = id2word[idx]
      embed = str(embeds[idx])
      fo.write(word+' '+embed+'\n')



class word2vec:
    def __init__(self, inputfile, vocabulary_size=500, embedding_dim=50, epoch_num=5, batch_size=16, windows_size=5,neg_sample_num=10):
        self.embedding_dim = embedding_dim
        self.windows_size = windows_size
        self.vocabulary_size = vocabulary_size
        self.op = Options(inputfile, self.vocabulary_size)
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.neg_sample_num = neg_sample_num
        self.embeds = {}
        self.inputfile = inputfile

    def vis(self, embeddings, testwords):
        class_words = self.op.vocab_words
        for w in testwords:
            for kc, vc in class_words.items():
                if vc == w:
                    print(vc + ": ")
                    print(embeddings[kc])

    def train(self):
        model = skipgram(self.vocabulary_size, self.embedding_dim)
        if torch.cuda.is_available():
          model.cuda()
        optimizer = optim.SGD(model.parameters(),lr=0.2)
        for epoch in range(self.epoch_num):
          start = time.time()
          self.op.process = True
          batch_num = 0
          batch_new = 0

          while self.op.process:
            pos_u, pos_v, neg_v = self.op.generate_batch(self.windows_size, self.batch_size, self.neg_sample_num)

            pos_u = Variable(torch.LongTensor(pos_u))
            pos_v = Variable(torch.LongTensor(pos_v))
            neg_v = Variable(torch.LongTensor(neg_v))


            if torch.cuda.is_available():
              pos_u = pos_u.cuda()
              pos_v = pos_v.cuda()
              neg_v = neg_v.cuda()

            optimizer.zero_grad()
            loss = model(pos_u, pos_v, neg_v,self.batch_size)

            loss.backward()

            optimizer.step()


            if batch_num%30000 == 0:
              torch.save(model.state_dict(), './tmp/skipgram.epoch{}.batch{}'.format(epoch,batch_num))

              if batch_num%2000 == 0:
                end = time.time()
                word_embeddings = model.input_embeddings()
                #self.vis(word_embeddings, testwords)
                batch_new = batch_num
                start = time.time()
            batch_num = batch_num + 1
        model.save_embedding("./article_test_embeds_short/embeds_" + os.path.basename(self.inputfile), self.op.vocab_words)


    # step 1: select a set of common words and use these as the vectors
    # analogue to PCA with words?



root = "./year_articles_test/"
texts = glob.glob(root + '*.txt')
testfile = 'sense.txt'
compfiles = ['mobydick.txt', 'emma.txt', 'annegg.txt', 'sense.txt']

print(texts)

if __name__ == '__main__':
    class_embeds = {}
    class_words = {}
    for f in texts:
        print(f + ": ")
        wc = word2vec(f)
        wc.train()
        print(f + " embedded")
        print("------------------")

if False:
    cumdifs = {}
    cumcounts = {}
    for f in compfiles[:-1]:
        cumdifs[f] = 0
        cumcounts[f] = 0
    for kt, vt in class_words[testfile].items():
        for f in compfiles[:-1]:
            for kc, vc in class_words[f].items():
                if vt == vc:
                    e0 = class_embeds[testfile][kt]
                    e1 = class_embeds[f][kc]
                    cumdifs[f] += np.linalg.norm(e0 - e1)
                    cumcounts[f] += 1

    for key, val in cumdifs.items():
        print("Cumulative difference " + key + ":")
        print(val / cumcounts[key])
