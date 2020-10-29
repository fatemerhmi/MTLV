from skmultilearn.model_selection import IterativeStratification
from prettytable import PrettyTable
import os

import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
import torch
import nltk
from nltk.corpus import reuters
from itertools import islice


from mtl.datasets.utils import iterative_train_test_split, create_dataLoader, create_new_column
import mtl.utils.logger as mlflowLogger 

def _setup_datasets(dataset_name, dataset_args, tokenizer, tokenizer_args, head_type, head_args, batch_size):
    #------- load or download then set up reuters dataset
    DATA_DIR = dataset_args['root']

    if os.path.exists(f"{DATA_DIR}/reuters"):
        #--------load dataframe
        print("[  dataset  ] reuters directory already exists.")
        train_val_df = pd.read_csv(f"{DATA_DIR}/{dataset_args['data_path']}/reuters_train.csv'")
        train_val_df['labels'] = train_val_df.apply(lambda row: np.array(ast.literal_eval(row['labels'])), axis=1)

        test_df = pd.read_csv(f"{DATA_DIR}/{dataset_args['data_path']}/reuters_test.csv'")
        test_df['labels'] = test_df.apply(lambda row: np.array(ast.literal_eval(row['labels'])), axis=1)

        #--------loading and storing labels to mlflow
        labels = np.load(f"{DATA_DIR}/reuters/labels.npy")
        num_labels = len(labels)
        mlflowLogger.store_param("col_names", labels)
        mlflowLogger.store_param("num_labels", num_labels)

    else:
        os.makedirs(f"{DATA_DIR}/reuters")
        nltk.download('reuters')

        train_val_df = pd.DataFrame(columns=["id", "text", "labels"])
        test_df = pd.DataFrame(columns=["id", "text", "labels"])

        documents = reuters.fileids()
        for doc_id in tqdm(documents):
            if doc_id.startswith('training/'):
                train_val_df = train_val_df.append({'id': doc_id, 'text': reuters.raw(doc_id), 'labels': reuters.categories(doc_id)}, ignore_index=True)
            elif doc_id.startswith('test/'):
                test_df = test_df.append({'id': doc_id, 'text': reuters.raw(doc_id), 'labels': reuters.categories(doc_id)}, ignore_index=True)

        labels = reuters.categories()

        for label in tqdm(labels):
            train_val_df = create_new_column(train_val_df, label)
            test_df = create_new_column(test_df, label)

        train_val_df.drop(['labels'],inplace=True, axis=1)
        test_df.drop(['labels'],inplace=True, axis=1)

        label_counts={}
        for label in labels:
            label_counts['label'] = train_val_df[label].sum()

        label_counts={}
        for label in labels:
            label_counts[label] = train_val_df[label].sum()
        label_counts = {k: v for k, v in sorted(label_counts.items(), key=lambda item: item[1], reverse=True)}
        top20_labels = list(islice(label_counts, 20))
        other_labels = l3 = [x for x in list(label_counts.keys()) if x not in top20_labels]

        train_val_df.drop(other_labels,inplace=True, axis=1)
        test_df.drop(other_labels,inplace=True, axis=1)

        labels = top20_labels

        np.save(f"{DATA_DIR}/reuters/labels.npy", labels)
        num_labels = len(labels)
        mlflowLogger.store_param("col_names", labels)
        mlflowLogger.store_param("num_labels", num_labels)

        train_val_df['labels'] = train_val_df.apply(lambda row: np.array(row[labels].to_list()), axis=1)
        test_df['labels'] = test_df.apply(lambda row: np.array(row[labels].to_list()), axis=1)

        train_val_df.drop(labels, inplace=True, axis=1)
        test_df.drop(labels, inplace=True, axis=1)

        #-------save the datafarme
        train_val_df_tosave = train_val_df.copy()
        train_val_df_tosave['labels'] = train_val_df_tosave.apply(lambda row: list(row["labels"]), axis=1)
        train_val_df_tosave.to_csv(f'{DATA_DIR}/reuters/reuters_train.csv', index=False)
        del train_val_df_tosave

        test_df_tosave = test_df.copy()
        test_df_tosave['labels'] = test_df_tosave.apply(lambda row: list(row["labels"]), axis=1)
        test_df_tosave.to_csv(f'{DATA_DIR}/reuters/reuters_test.csv', index=False)
        del test_df_tosave

    #-------shuffle
    train_val_df = train_val_df.sample(frac=1).reset_index(drop=True)
    mlflowLogger.store_param("dataset.len", len(train_val_df)+ len(test_df))

    #-------stratified sampling
    train_indexes, val_indexes = iterative_train_test_split(train_val_df['text'], np.array(train_val_df['labels'].to_list()), 0.15)

    train_df = train_val_df.iloc[train_indexes,:]
    val_df = train_val_df.iloc[val_indexes,:]

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    print('Train: ', len(train_df))
    mlflowLogger.store_param("dataset.train.len", len(train_df))
    print('Test: ', len(test_df))
    mlflowLogger.store_param("dataset.test.len", len(test_df))
    print('Val: ', len(val_df))
    mlflowLogger.store_param("dataset.val.len", len(val_df))

    #-------table for train, test, val counts
    label_counts_train = np.array(train_df.labels.to_list()).sum(axis=0)
    label_counts_test = np.array(test_df.labels.to_list()).sum(axis=0)
    label_counts_val = np.array(val_df.labels.to_list()).sum(axis=0)


    pretty=PrettyTable()
    label_counts_total = []
    pretty.field_names = ['Label', 'total', 'train', 'test','val']
    for pathology, cnt_train, cnt_test, cnt_val in zip(labels, label_counts_train, label_counts_test, label_counts_val):
        cnt_total = cnt_train + cnt_test + cnt_val
        label_counts_total.append(cnt_total)
        pretty.add_row([pathology, cnt_total, cnt_train, cnt_test, cnt_val])
    print(pretty)

    #-------check for multi-head, single or multi-task
    if head_type =="multi-task":
        #check the type:   label_counts_total
        if head_args['type'] == "givenset":
            heads_index = head_args["heads_index"]

        elif head_args['type'] == "KDE":
            print("[  dataset  ] KDE label grouping starts!")
            heads_index = KDE(label_counts_total, head_args['bandwidth'])

        elif head_args['type'] == "meanshift":
            print("[  dataset  ] meanshift label grouping starts!")
            heads_index = meanshift(label_counts_total)

        elif head_args['type'] == "kmediod-label":
            print("[  dataset  ] kmediod-label grouping starts!")
            model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
            embds = get_all_label_embds(labels, tokenizer, model)
            if "elbow" in head_args.keys():
                plot_elbow_method(embds,head_args['elbow'])
            heads_index = grouping_kmediod(embds, head_args['clusters'])
            del model

        elif head_args['type'] == "kmediod-labeldesc":
            print("[  dataset  ] kmediod-label description grouping starts!")
            model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
            labels_list = [labels_dict[label] for label in labels]
            # list(labels_dict.values()
            embds = get_all_label_embds(labels_list, tokenizer, model)
            if "elbow" in head_args.keys():
                plot_elbow_method(embds,head_args['elbow'])
            heads_index = grouping_kmediod(embds, head_args['clusters'])
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

    elif head_type =="single-head":

        #--prepare labels for dataloader
        train_labels = torch.from_numpy(np.array(train_df.labels.to_list()))
        test_labels = torch.from_numpy(np.array(test_df.labels.to_list()))
        val_labels = torch.from_numpy(np.array(val_df.labels.to_list()))
    else:
        raise Exception("The head type must be either 'multi-task' or 'single-head'!")

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

labels_dict = {
    'earn': "Earnings and Earnings Forecasts ", 
    'acq': "Mergers Acquisitions ", 
    'money-fx': "Money Foreign Exchange ", 
    'grain': "grain", 
    'crude': "crude", 
    'trade': "trade", 
    'interest': "interest", 
    'wheat': "wheat", 
    'ship': "ship", 
    'corn': "corn", 
    'money-supply': "money-supply", 
    'dlr': "dlr", 
    'sugar': "sugar", 
    'oilseed': "oilseed", 
    'coffee': "coffee", 
    'gnp': "gnp", 
    'gold': "gold", 
    'veg-oil': "Vegetable oil", 
    'soybean': "soybean", 
    'bop': "bop",
}