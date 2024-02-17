from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle
import numpy as np
import random


def showEmbeddingTGN(content_vec, labels):
    tsne_content = PCA(n_components=2)
    result_content = tsne_content.fit_transform(content_vec, labels)
    idx = np.where(result_content[:, 0] < 0.27)[0]
    result_content = result_content[idx]
    labels = np.array(labels)[idx]

    idx = np.where(result_content[:, 1] < 100)[0]
    result_content = result_content[idx]
    labels = np.array(labels)[idx]
    
    x_min, x_max = np.min(result_content[:, 0], 0), np.max(result_content[:, 0], 0)
    y_min, y_max = np.min(result_content[:, 1], 0), np.max(result_content[:, 1], 0)
    result_content[:, 0] = (result_content[:, 0] - x_min) / (x_max - x_min)
    result_content[:, 1] = (result_content[:, 1] - y_min) / (y_max - y_min)

    idx = np.where(result_content[:, 0] < 0.25)[0]
    result_content = result_content[idx]
    labels = np.array(labels)[idx]


    # idx = np.where(result_content[:, 1] < 0.6)[0]
    # result_content = result_content[idx]
    # labels = np.array(labels)[idx]
    fig1 = plot_embedding_2D(result_content, labels, 'TGN')
    # plt.show()
    plt.savefig('./tsne/xinyongka_tgn.png', dpi=600)

def showEmbeddingTGAT(content_vec, labels):
    tsne_content = PCA(n_components=2)
    result_content = tsne_content.fit_transform(content_vec, labels)
    idx = np.where(result_content[:, 0] < 0.25)[0]
    result_content = result_content[idx]
    labels = np.array(labels)[idx]

    idx = np.where(result_content[:, 0] > -0.02)[0]
    result_content = result_content[idx]
    labels = np.array(labels)[idx]

    idx = np.where(result_content[:, 1] < 0.05)[0]
    result_content = result_content[idx]
    labels = np.array(labels)[idx]

    fig1 = plot_embedding_2D(result_content, labels, 'TGAT')
    # plt.show()
    plt.savefig('./tsne/xinyongka_tgat.png', dpi=600)


def showEmbeddingDyTES(content_vec, labels):
    tsne_content = PCA(n_components=2)
    result_content = tsne_content.fit_transform(content_vec[:10000], labels[:10000])
    idx = np.where(result_content[:, 1] < 0.008)[0]
    result_content = result_content[idx]
    labels = np.array(labels)[idx]
    fig1 = plot_embedding_2D(result_content, labels, 'DyTES')
    # plt.show()
    plt.savefig('./tsne/xinyongka_tgat.png', dpi=600)


def plot_embedding_2D(data, label, title):

    fig = plt.figure()
    
    for i in tqdm(range(data.shape[0])):
        if label[i] == 0:
            if random.random() > 0.25:
                plt.scatter(data[i, 0], data[i, 1], c='b', s=1)
    n = 0
    for i in tqdm(range(data.shape[0])):
        if label[i] == 1:
            plt.scatter(data[i, 0], data[i, 1], c='r', s=1)
            n += 1
    print(n)

    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


if __name__ == '__main__':
    # with open('./tsne/xinyongka_labels.pkl', 'rb') as f:
    #     labels = pickle.load(f)
    # with open('./tsne/xinyongka.pkl', 'rb') as f:
    #     d = pickle.load(f)

    # with open('./tsne/xinyongka_tsne_tgat.pkl', 'rb') as f:
    #     d = pickle.load(f)
    
    # x = d['x'].cpu()
    # showEmbedding(x, d['l'])

    with open('./tsne/xinyongka_tsne_tgn.pkl', 'rb') as f:
        d = pickle.load(f)
    
    x = d['x'][-21000:]
    showEmbeddingTGN(x, d['l'][-21000:])




