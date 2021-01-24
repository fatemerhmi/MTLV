import numpy as np

#TODO gouping techniques

__all__ = ["group_heads", "padding_heads"]

def padding_heads(heads_index): #as heads does not have the same size, we pad them with -1
    head_counts = [len(head) for head in heads_index]
    lenmax = max(head_counts)
    padded_heads = []
    for head in heads_index:
        pad_n = lenmax - len(head)
        arr = np.pad(head, (0,pad_n), 'constant', constant_values=-1)
        padded_heads.append(arr)
    return padded_heads

def reorder(row,groups_index):
#     print(row)
    groups_values = []
    for group in groups_index:
        tmp =[]
        for index in group:
            if index==-1:
                tmp.append(-1)
            else:
                tmp.append(row[index])
        groups_values.append(tmp)
#     print(groups_values)
    return groups_values

def group_heads(groups,df): # first attempt for each label one head
    df.loc[:,'head_labels'] = df.apply(lambda row: reorder(row['labels'],groups), axis=1)
    return df

def pyMeshSim():

    
    return sim