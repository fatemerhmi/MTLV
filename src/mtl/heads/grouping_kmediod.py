import matplotlib
import seaborn as sns
import pandas as pd
# from sklearn.datasets import load_digits

import torch
from scipy import spatial
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import mtl.utils.logger as mlflowLogger 

def cos_sim(vec1, vec2):
    return 1 - spatial.distance.cosine(vec1, vec2)

def get_embd(word, tokenizer, model):
    inputs = tokenizer(word, return_tensors="pt")
    outputs = model(**inputs)
    hidden_units = outputs.last_hidden_state[0]
    # print(torch.mean(hidden_units,axis=0).shape)
    return torch.mean(hidden_units,axis=0)

def get_all_label_embds(labels_title, tokenizer, model):
    embds = []
    for label in labels_title:
        embds.append(get_embd(label, tokenizer, model).detach().numpy())
    embds = np.array(embds)
    embds.shape

    mms = MinMaxScaler()
    mms.fit(embds)
    data_transformed = mms.transform(embds)

    return data_transformed

def plot_elbow_method(data, Krange):
    Sum_of_squared_distances = []
    K = range(1,Krange)
    for k in K:
        km = KMedoids(n_clusters=k)
        km = km.fit(data)
        Sum_of_squared_distances.append(km.inertia_) # inertia is calculated as the sum of squared distance for each point to it's closest centroid

    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    mlflowLogger.store_pic(plt, 'Elbow_method', 'png')
    # plt.savefig(f'elbow.png', dpi=100)

def grouping_kmediod(data, n_clusters_, labels_title):
    # n_clusters_ = 4
    kmedoids = KMedoids(n_clusters=n_clusters_, random_state=0).fit(data)
    heads_index = [list(np.where(kmedoids.labels_==i)[0]) for i in range(n_clusters_)]

    #------printing the groups
    groupings = []
    for head_index in heads_index:
        tmp = []
        for i in head_index:
            tmp.append(labels_title[i])
        groupings.append(tmp)
    mlflowLogger.store_param("groups_label", groupings)
    # for group in groupings:
    #     print(group)
    
    return heads_index, kmedoids.labels_

def plot_emb_groups(embds, labels, cluster_label):

    # digits = load_digits()

    # data_X = embds
    # y = cluster_label

    from sklearn.manifold import TSNE
    # tsne = TSNE(n_components=2, random_state=0, n_iter = 3000, perplexity=5)

    # tsne_obj= tsne.fit_transform(data_X)

    # tsne_df = pd.DataFrame({'X':tsne_obj[:,0],
    #                         'Y':tsne_obj[:,1],
    #                         'cluster':y})

    # tsne_plot = sns.scatterplot(x="X", y="Y",
    #           hue="cluster",
    #           palette=None,
    #           legend='full',
    #           data=tsne_df);            

    # LABEL_COLOR_MAP = {0 : 'r',
    #                1 : 'k',
    #                ....,
    #                }

    # label_color = [LABEL_COLOR_MAP[l] for l in labels]

    # from sklearn.decomposition import PCA
    # pca = PCA(n_components = 20, random_state = 7)
    # embds = pca.fit_transform(embds)
    
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis

    # tsne = TSNE(n_components=2, random_state=0, n_iter = 3000, perplexity=5)
    tsne = TSNE(n_components = 2, perplexity = 10, random_state = 6, 
                learning_rate = 1000, n_iter = 1500)
    
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(embds)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    
    # display scatter plot
    ax.scatter(x_coords, y_coords, c = cluster_label)

    for label, x, y in zip(labels, x_coords, y_coords):
        ax.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=6, ha='center') #clip_on=True
    plt.xlim(x_coords.min()-(0.2*x_coords.max()), x_coords.max()+(0.2*x_coords.max()))
    plt.ylim(y_coords.min()-(0.2*x_coords.max()), y_coords.max()+(0.2*x_coords.max()))
    # plt.show()


    # fig = tsne_plot.get_figure()
    plt.title('TSNE of embds')
    plt.xlabel('X')
    plt.ylabel('Y')
    mlflowLogger.store_pic(fig, 'tsne', 'png')


    
    # ax.plot([0,1,2], [10,20,3])
    # fig.savefig('path/to/save/image/to.png')   # save the figure to file
    # plt.close(fig) 