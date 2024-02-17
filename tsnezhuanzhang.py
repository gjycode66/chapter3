from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle
import numpy as np
import random


def showEmbeddingDyTES(content_vec, labels):
    tsne_content = PCA(n_components=2)
    result_content = tsne_content.fit_transform(content_vec, labels)
    # idx = np.where(result_content[:, 0] < 0.25)[0]
    # result_content = result_content[idx]
    # labels = np.array(labels)[idx]

    # idx = np.where(result_content[:, 0] > -0.02)[0]
    # result_content = result_content[idx]
    # labels = np.array(labels)[idx]

    idx = np.where(result_content[:, 1] < 0.1)[0]
    result_content = result_content[idx]
    labels = np.array(labels)[idx]

    idx = np.where(result_content[:, 1] > -0.04)[0]
    result_content = result_content[idx]
    labels = np.array(labels)[idx]

    fig1 = plot_embedding_2D(result_content, labels, 'DyTES')
    # plt.show()
    plt.savefig('./tsne/zhuanzhang_dytes.png', dpi=600)


def showEmbeddingTGAT(content_vec, labels):
    tsne_content = PCA(n_components=2)
    result_content = tsne_content.fit_transform(content_vec[: 13000], labels[: 13000])
    idx = np.where(result_content[:, 1] < 5.5)[0]
    result_content = result_content[idx]
    labels = np.array(labels)[idx]
    fig1 = plot_embedding_2D(result_content, labels, 'TGAT')
    # plt.show()
    plt.savefig('./tsne/zhuanzhang_tgat.png', dpi=600)


def plot_embedding_2D(data, label, title):

    fig = plt.figure()
    
    for i in tqdm(range(data.shape[0])):
        if label[i] == 0:
            plt.scatter(data[i, 0], data[i, 1], c='b', s=1)
    n = 0
    for i in tqdm(range(data.shape[0])):
        if label[i] == 1:
            if random.random() > 0.80:
                plt.scatter(data[i, 0], data[i, 1], c='r', s=1)
                n += 1
    print(n)

    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


if __name__ == '__main__':
    # with open('./tsne/zhuanzhang_labels.pkl', 'rb') as f:
    #     labels = pickle.load(f)
    # with open('./tsne/zhuanzhang.pkl', 'rb') as f:
    #     d = pickle.load(f)
    # showEmbeddingDyTES(d['x'], np.array(labels['labels'], dtype='int'))
    with open('./tsne/zhuanzhang_tsne.pkl', 'rb') as f:
        d = pickle.load(f)
    
    x = d['x'].cpu()
    showEmbeddingTGAT(x, d['l'])

    # with open('./tsne/xinyongka_tsne_tgn.pkl', 'rb') as f:
    #     d = pickle.load(f)
    
    # x = d['x'][-21000:]
    # showEmbeddingTGN(x, d['l'][-21000:])




