import hdbscan
import numpy as np

import mtl.utils.logger as mlflowLogger 

def hdbscan_clustering(data, label_names):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
    clusterer.fit(data)
    n_clusters = clusterer.labels_.max()
    heads_index = [list(np.where(clusterer.labels_==i)[0]) for i in range(n_clusters)]

    #------printing the clusters
    groupings = []
    for head_index in heads_index:
        tmp = []
        for i in head_index:
            tmp.append(label_names[i])
        groupings.append(tmp)
    mlflowLogger.store_param("label_clusters", groupings)

    return heads_index, clusterer.labels_, n_clusters