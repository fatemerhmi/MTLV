import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

__all__ = ['meanshift_labelcounts']

def meanshift_labelcounts(x):
  """
  meanshift label counts function recieved the label counts as an input and returns the clustering in a two dimentional list.
  """
  X = np.reshape(x, (-1, 1))
  ms = MeanShift(bandwidth=None, bin_seeding=True)
  ms.fit(X)
  labels = ms.labels_
  cluster_centers = ms.cluster_centers_

  labels_unique = np.unique(labels)
  n_clusters_ = len(labels_unique)

  # print("number of estimated clusters : %d" % n_clusters_)
  # print(labels)
  heads_index = [list(np.where(labels==i)[0]) for i in range(n_clusters_)]
  # print(heads_index)
  # groupings = []
  # for head_index in heads_index:
  #   tmp = []
  #   for i in head_index:
  #     tmp.append(x[i])
  #   groupings.append(tmp)
  # print(groupings)
  return heads_index

