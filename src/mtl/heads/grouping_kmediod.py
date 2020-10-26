
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

def grouping_kmediod(data, n_clusters_):
    n_clusters_ = 4
    kmedoids = KMedoids(n_clusters=n_clusters_, random_state=0).fit(data)
    heads_index = [list(np.where(kmedoids.labels_==i)[0]) for i in range(n_clusters_)]

    #------printing the groups
    # groupings = []
    # for head_index in heads_index:
    #     tmp = []
    #     for i in head_index:
    #         tmp.append(labels_title[i])
    #     groupings.append(tmp)
    # for group in groupings:
    #     print(group)
    return heads_index