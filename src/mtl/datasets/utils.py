import os
import os.path
import hashlib
import errno
import tarfile
import torch
import requests
import csv
import torch
from tqdm import tqdm
import os
import re
import sys
import zipfile
import gzip
import pandas as pd
import ast
import numpy as np
from skmultilearn.model_selection import IterativeStratification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from prettytable import PrettyTable

from mtl.heads.KDE_clustering import *
from mtl.heads.meanshift_clustering import *
from mtl.heads.kmediod_clustering import *
from mtl.heads.HDBSCAN_clustering import hdbscan_clustering
import mtl.utils.configuration as configuration
from mtl.heads.utils import padding_heads, group_heads
import mtl.utils.logger as mlflowLogger 

def iterative_train_test_split(X, y, test_size):
    stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[test_size, 1.0-test_size])
    train_indexes, test_indexes = next(stratifier.split(X, y))

    return train_indexes, test_indexes

def create_dataLoader(input, labels, batch_size):
    data = TensorDataset(input.input_ids, input.attention_mask, labels)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader

def create_new_column(df, column_name):
    df[column_name] = df.apply(lambda row: \
                                        1 if column_name in row['labels'] \
                                        else 0, \
                                        axis=1)
    return df

def preprocess(train_df, test_df, val_df, tokenizer, tokenizer_args, labels, labels_dict, head_type, head_args, num_labels, model_cfg, batch_size):

    print('Train: ', len(train_df))
    mlflowLogger.store_param("dataset.train.len", len(train_df))
    print('Test: ', len(test_df))
    mlflowLogger.store_param("dataset.test.len", len(test_df))
    print('Val: ', len(val_df))
    mlflowLogger.store_param("dataset.val.len", len(val_df))

    #------- table for train, test, val counts
    label_counts_train = np.array(train_df.labels.to_list()).sum(axis=0)
    label_counts_test = np.array(test_df.labels.to_list()).sum(axis=0)
    label_counts_val = np.array(val_df.labels.to_list()).sum(axis=0)

    pretty = PrettyTable()
    label_counts_total = []
    pretty.field_names = ['Label', 'total', 'train', 'test','val']
    for pathology, cnt_train, cnt_test, cnt_val in zip(labels, label_counts_train, label_counts_test, label_counts_val):
        cnt_total = cnt_train + cnt_test + cnt_val
        label_counts_total.append(cnt_total)
        pretty.add_row([pathology, cnt_total, cnt_train, cnt_test, cnt_val])
    print(pretty)
    
    #-------check for multi-head, single or multi-task
    if head_type =="MTL" or head_type =="GMTL":
        #check the type:   
        if head_args['type'] == "givenset":
            heads_index = head_args["heads_index"]

        elif head_args['type'] == "KDE":
            print("[  dataset  ] KDE label grouping starts!")
            heads_index = KDE(label_counts_train, head_args['bandwidth'])

        elif head_args['type'] == "meanshift":
            print("[  dataset  ] meanshift label grouping starts!")
            heads_index = meanshift_labelcounts(label_counts_train)

        elif head_args['type'] == "kmediod-label":
            print("[  dataset  ] kmediod-label grouping starts!")
            # model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
            model = configuration.setup_model(model_cfg)(num_labels, "emb_cls")
            embds = get_all_label_embds(labels, tokenizer, model)
            if "elbow" in head_args.keys():
                plot_elbow_method(embds,head_args['elbow'])
            heads_index, cluster_label = kmediod_clustering(embds, head_args['clusters'], labels)
            if "plot" in head_args.keys():
                if head_args['plot'] == True:
                    plot_emb_tsne_clusters(embds, labels, cluster_label)
            del model

        elif head_args['type'] == "kmediod-labeldesc":
            print("[  dataset  ] kmediod-label description grouping starts!")
            # model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
            model = configuration.setup_model(model_cfg)(num_labels, "emb_cls")
            labels_description_list = [labels_dict[label] for label in labels]
            # list(labels_dict.values()
            embds = get_all_label_embds(labels_description_list, tokenizer, model)
            if "elbow" in head_args.keys():
                plot_elbow_method(embds,head_args['elbow'])
            n_cluster = head_args['clusters']
            heads_index, cluster_label = kmediod_clustering(embds, n_cluster, labels)
            if "plot" in head_args.keys():
                if head_args['plot'] == True:
                    plot_emb_tsne_clusters(embds, labels, cluster_label)
            del model

        mlflowLogger.store_param("heads_index", heads_index)
        padded_heads = padding_heads(heads_index)
        
        #--group the heads
        train_df = group_heads(padded_heads, train_df)
        test_df = group_heads(padded_heads, test_df)
        val_df = group_heads(padded_heads, val_df)

        #--prepare labels for dataloader
        train_labels = torch.from_numpy(np.array(train_df.head_labels.to_list()))
        test_labels = torch.from_numpy(np.array(test_df.head_labels.to_list()))
        val_labels = torch.from_numpy(np.array(val_df.head_labels.to_list()))

    elif head_type =="single-head" or head_type == "STL_cls":
        #--prepare labels for dataloader
        train_labels = torch.from_numpy(np.array(train_df.labels.to_list()))
        test_labels = torch.from_numpy(np.array(test_df.labels.to_list()))
        val_labels = torch.from_numpy(np.array(val_df.labels.to_list()))

    else:
        raise Exception(f"The head type must be either 'MTL' or 'single-head' or 'STL', you entered {head_type}!")


    #-------tokenize
    reports_train = train_df.text.to_list()
    reports_test = test_df.text.to_list()
    reports_val   = val_df.text.to_list()

    train = tokenizer(reports_train, \
                        padding=tokenizer_args['padding'], \
                        truncation=tokenizer_args['truncation'], \
                        max_length=tokenizer_args['max_length'], \
                        return_tensors="pt")

    test = tokenizer(reports_test, \
                        padding=tokenizer_args['padding'], \
                        truncation=tokenizer_args['truncation'], \
                        max_length=tokenizer_args['max_length'], \
                        return_tensors="pt")
    val = tokenizer(reports_val, \
                        padding=tokenizer_args['padding'], \
                        truncation=tokenizer_args['truncation'], \
                        max_length=tokenizer_args['max_length'], \
                        return_tensors="pt")
    
    #-------create dataloarders
    train_dataloader      = create_dataLoader(train, train_labels, batch_size)
    validation_dataloader = create_dataLoader(val, val_labels, batch_size)
    test_dataloader       = create_dataLoader(test, test_labels, batch_size)

    return train_dataloader, validation_dataloader, test_dataloader, num_labels

def preprocess_cv(train_df, test_df, val_df, tokenizer, tokenizer_args, labels, labels_dict, head_type, head_args, num_labels, model_cfg, batch_size, fold_i):
    print('Train: ', len(train_df))
    mlflowLogger.store_param(f"dataset.train.len.Fold{fold_i}", len(train_df))
    print('Test: ', len(test_df))
    mlflowLogger.store_param(f"dataset.test.len.Fold{fold_i}", len(test_df))
    print('Val: ', len(val_df))
    mlflowLogger.store_param(f"dataset.val.len.Fold{fold_i}", len(val_df))

    #-------table for train, test,val counts      
    label_counts_train = np.array(train_df.labels.to_list()).sum(axis=0)
    label_counts_test = np.array(test_df.labels.to_list()).sum(axis=0)
    label_counts_val = np.array(val_df.labels.to_list()).sum(axis=0)

    pretty=PrettyTable()
    label_counts_total = []
    pretty.field_names = ['Pathology', 'total', 'train', 'test','val']
    for pathology, cnt_train, cnt_test, cnt_val in zip(labels, label_counts_train, label_counts_test, label_counts_val):
        cnt_total = cnt_train + cnt_test + cnt_val
        label_counts_total.append(cnt_total)
        pretty.add_row([pathology, cnt_total, cnt_train, cnt_test, cnt_val])
    print(pretty)

    #-------check for head types: STL, MTL, GMTL, GMHL
    if head_type =="GMTL" or head_type =="GMHL":
        #check the type:   
        if head_args['type'] == "givenset":
            heads_index = head_args["heads_index"]

        elif head_args['type'] == "KDE":
            print("[  dataset  ] KDE label clustering starts!")
            heads_index = KDE(label_counts_train, head_args['bandwidth'])

        elif head_args['type'] == "meanshift":
            print("[  dataset  ] meanshift label clustering starts!")
            heads_index = meanshift_labelcounts(label_counts_train)

        elif head_args['type'] == "kmediod-label":
            print("[  dataset  ] kmediod-label clustering starts!")
            model = configuration.setup_model(model_cfg)(num_labels, "emb_cls")
            embds = get_all_label_embds(labels, tokenizer, model)
            if "elbow" in head_args.keys():
                plot_elbow_method(embds,head_args['elbow'])
            heads_index, cluster_label = kmediod_clustering(embds, head_args['clusters'], labels)
            if "plot" in head_args.keys():
                if head_args['plot'] == True:
                    plot_emb_tsne_clusters(embds, labels, cluster_label)
            del model

        elif head_args['type'] == "kmediod-labeldesc":
            print("[  dataset  ] kmediod-label description clustering starts!")
            model = configuration.setup_model(model_cfg)(num_labels, "emb_cls")
            labels_description_list = [labels_dict[label] for label in labels]
            embds = get_all_label_embds(labels_description_list, tokenizer, model)
            if "elbow" in head_args.keys():
                plot_elbow_method(embds,head_args['elbow'])
            n_cluster = head_args['clusters']
            heads_index, cluster_label = kmediod_clustering(embds, n_cluster, labels)
            if "plot" in head_args.keys():
                if head_args['plot'] == True:
                    plot_emb_tsne_clusters(embds, labels, cluster_label)
            del model

        elif head_args['type'] == "hdbscan":
            print("[  dataset  ] HDBSCAN label clustering starts!")
            model = configuration.setup_model(model_cfg)(num_labels, "emb_cls")
            embds = get_all_label_embds(labels, tokenizer, model)
            if "elbow" in head_args.keys():
                Exception("[dataset] No elbow method for hdbscan!")
            heads_index, cluster_label, n_clusters = hdbscan_clustering(embds, labels)
            if "plot" in head_args.keys():
                if head_args['plot'] == True:
                    plot_emb_tsne_clusters(embds, labels, cluster_label)
            del model

        elif head_args['type'] == "hdbscan-labeldesc":
            print("[  dataset  ] HDBSCAN label clustering starts!")
            model = configuration.setup_model(model_cfg)(num_labels, "emb_cls")
            labels_description_list = [labels_dict[label] for label in labels]
            embds = get_all_label_embds(labels_description_list, tokenizer, model)
            if "elbow" in head_args.keys():
                Exception("[dataset] No elbow method for hdbscan!")
            heads_index, cluster_label, n_clusters = hdbscan_clustering(embds, labels)
            if "plot" in head_args.keys():
                if head_args['plot'] == True:
                    plot_emb_tsne_clusters(embds, labels, cluster_label)
            del model

        mlflowLogger.store_param(f"heads_index", heads_index)
        padded_heads = padding_heads(heads_index)
        
        #--group the heads
        train_df = group_heads(padded_heads, train_df)
        test_df = group_heads(padded_heads, test_df)
        val_df = group_heads(padded_heads, val_df)

        #--prepare labels for dataloader
        train_labels = torch.from_numpy(np.array(train_df.head_labels.to_list()))
        test_labels = torch.from_numpy(np.array(test_df.head_labels.to_list()))
        val_labels = torch.from_numpy(np.array(val_df.head_labels.to_list()))

    elif (head_type =="MTL") or (head_type == "STL"):
        #--prepare labels for dataloader
        train_labels = torch.from_numpy(np.array(train_df.labels.to_list()))
        test_labels = torch.from_numpy(np.array(test_df.labels.to_list()))
        val_labels = torch.from_numpy(np.array(val_df.labels.to_list()))
    else:
        raise Exception(f"The head type must be either 'MTL' or 'single-head' or 'STL', you entered {head_type}!")

    #-------tokenize
    reports_train = train_df.text.to_list()
    reports_test = test_df.text.to_list()
    reports_val   = val_df.text.to_list()

    train = tokenizer(reports_train, padding=tokenizer_args['padding'], truncation=tokenizer_args['truncation'], max_length=tokenizer_args['max_length'], return_tensors="pt")
    test = tokenizer(reports_test, padding=tokenizer_args['padding'], truncation=tokenizer_args['truncation'], max_length=tokenizer_args['max_length'], return_tensors="pt")
    val = tokenizer(reports_val, padding=tokenizer_args['padding'], truncation=tokenizer_args['truncation'], max_length=tokenizer_args['max_length'], return_tensors="pt")

    #-------create dataloarders
    train_dataloader      = create_dataLoader(train, train_labels, batch_size)
    validation_dataloader = create_dataLoader(val, val_labels, batch_size)
    test_dataloader       = create_dataLoader(test, test_labels, batch_size)

    return train_dataloader, validation_dataloader, test_dataloader, num_labels

def save_fold_train_validation(train_df, val_df, dataset_args, fold_i):
    DATA_DIR = dataset_args['root']

    train_df['labels'] = train_df.apply(lambda row: list(row["labels"]), axis=1)
    train_df.to_csv(f"{DATA_DIR}/{dataset_args['data_path']}/train_fold{fold_i}.csv", index=False)
    
    val_df['labels'] = val_df.apply(lambda row: list(row["labels"]), axis=1)
    val_df.to_csv(f"{DATA_DIR}/{dataset_args['data_path']}/validation_fold{fold_i}.csv", index=False)

def read_fold_train_validattion(fold_i, dataset_args):
    DATA_DIR = dataset_args['root']

    #---------load dataframe
    print("[  dataset  ] reading Fold:{fold_i} train and validation set.")
    
    #--------load dataframe
    train_df = pd.read_csv(f"{DATA_DIR}/{dataset_args['data_path']}/train_fold{fold_i}.csv")
    val_df = pd.read_csv(f"{DATA_DIR}/{dataset_args['data_path']}/validation_fold{fold_i}.csv")

    train_df.replace(np.nan, "", inplace=True)
    val_df.replace(np.nan, "", inplace=True)

    train_df.loc[:,'labels'] = train_df.labels.apply(ast.literal_eval)
    val_df.loc[:,'labels'] = val_df.labels.apply(ast.literal_eval)

    return train_df, val_df