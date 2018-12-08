from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import glob

# compare distances between analogy words?

def visualizer(embedfiles):
    testwords = ['man', 'woman', 'rich', 'poor', 'horse', 'land', 'house', 'sea']
    anchorwords = ['good', 'bad', 'went', 'came']
    preanmat = []
    namedembeds = {}
    matrices = {}
    plt.figure(10)
    totalbasis = []
    count = 1
    for embedfile in embedfiles:
        print("file")
        print(embedfile)
        fileembeds = {}
        with open(embedfile) as file:
            sepd = file.read().split(']')
            for line in sepd[:-1]:
                line = line.replace('[','')
                comps = line.split()
                if comps[0] in testwords or comps[0] in anchorwords:
                    embvec = [float(f) for f in comps[1:]]
                    fileembeds[comps[0]] = embvec
        labels = fileembeds.keys()
        matrix = []
        for l in labels:
            matrix.append(fileembeds[l])
        matrix = np.array(list(matrix))
        pca = PCA(n_components=2)
        pca.fit(matrix)
        trans = pca.transform(matrix)
        plt.subplot(2, 5, count)
        colors = ['r' if l in testwords else 'g' for l in labels]
        plt.scatter(trans[:,0], trans[:,1], marker = 'o', color=colors)
        for i, w in enumerate(labels):
            plt.annotate(
                w,
                xy = (trans[:,0][i], trans[:,1][i]), xytext = (3, 3),
                textcoords = 'offset points', ha = 'left', va = 'top')
        if False:
            anmat = []
            outlabs = 0
            for aword in anchorwords:
                anmat.append(fileembeds[aword])
            anmat = np.array(anmat)
            if preanmat == []:
                preanmat = anmat
                A = np.identity(len(anmat[0]))
            else:
                A, res, rank, s = np.linalg.lstsq(anmat, preanmat)

        matrix = []
        for l in labels:
            #transembed = np.dot(np.array(fileembeds[l]), A)
            transembed = fileembeds[l]
            matrix.append(transembed)
            totalbasis.append(transembed)
        matrix = np.array(list(matrix))
        matrices[embedfile] = (labels, matrix)
        count += 1

    colpca = PCA(n_components=2)
    colpca.fit(totalbasis)
    count = 1
    newmat = []
    cont = True

    for fname, tup in matrices.items():
        labels = list(tup[0])
        for word in anchorwords:
            if word not in labels:
                cont = False
        if cont:
            matrix = tup[1]
            newmatrix = np.array(list(matrix))
            trans = colpca.transform(newmatrix)

            oldmat = []

            for word in anchorwords:
                ind = labels.index(word)
                oldmat.append(trans[ind])
            if newmat == []:
                newmat = oldmat
            pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
            unpad = lambda x: x[:,:-1]
            A, res, rank, s = np.linalg.lstsq(pad(np.array(oldmat)), pad(np.array(newmat)))

            trans2 = np.dot(pad(trans), A)
            trans3 = unpad(trans2)
            plt.subplot(2, 5, count + 5)
            colors = ['r' if l in testwords else 'g' for l in labels]
            plt.scatter(trans3[:,0], trans3[:,1], marker = 'o', color=colors)
            for i, w in enumerate(labels):
                plt.annotate(
                    w,
                    xy = (trans3[:,0][i], trans3[:,1][i]), xytext = (3, 3),
                    textcoords = 'offset points', ha = 'left', va = 'top')
            count += 1



    plt.show()

root = "./article_embeds/"
texts = glob.glob(root + '*.txt')
visualizer(texts[:5])
