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


from mtl.datasets.utils import iterative_train_test_split, create_dataLoader, create_new_column, preprocess, preprocess_cv, save_fold_train_validation, read_fold_train_validattion
import mtl.utils.logger as mlflowLogger 

def reuters_dataset_preprocess(dataset_args):
    #------- load or download then set up reuters dataset
    DATA_DIR = dataset_args['root']

    if os.path.exists(f"{DATA_DIR}/reuters"):
        #--------load dataframe
        print("[  dataset  ] reuters directory already exists.")
        train_df_orig = pd.read_csv(f"{DATA_DIR}/{dataset_args['data_path']}/reuters_train.csv")
        train_df_orig['labels'] = train_df_orig.apply(lambda row: np.array(ast.literal_eval(row['labels'])), axis=1)

        test_df = pd.read_csv(f"{DATA_DIR}/{dataset_args['data_path']}/reuters_test.csv")
        test_df['labels'] = test_df.apply(lambda row: np.array(ast.literal_eval(row['labels'])), axis=1)

        mlflowLogger.store_param("dataset.len", len(train_df_orig)+len(test_df))

        #--------loading and storing labels to mlflow
        labels = np.load(f"{DATA_DIR}/{dataset_args['data_path']}/labels.npy")
        labels = list(labels)
        num_labels = len(labels)
        mlflowLogger.store_param("col_names", labels)
        mlflowLogger.store_param("num_labels", num_labels)

        return train_df_orig, test_df, labels, num_labels

    else:
        os.makedirs(f"{DATA_DIR}/{dataset_args['data_path']}")
        nltk.download('reuters')

        train_df_orig = pd.DataFrame(columns=["id", "text", "labels"])
        test_df = pd.DataFrame(columns=["id", "text", "labels"])

        documents = reuters.fileids()
        for doc_id in tqdm(documents):
            if doc_id.startswith('training/'):
                train_df_orig = train_df_orig.append({'id': doc_id, 'text': reuters.raw(doc_id), 'labels': reuters.categories(doc_id)}, ignore_index=True)
            elif doc_id.startswith('test/'):
                test_df = test_df.append({'id': doc_id, 'text': reuters.raw(doc_id), 'labels': reuters.categories(doc_id)}, ignore_index=True)

        labels = reuters.categories()

        for label in tqdm(labels):
            train_df_orig = create_new_column(train_df_orig, label)
            test_df = create_new_column(test_df, label)

        train_df_orig.drop(['labels'],inplace=True, axis=1)
        test_df.drop(['labels'],inplace=True, axis=1)

        label_counts={}
        for label in labels:
            label_counts['label'] = train_df_orig[label].sum()

        label_counts={}
        for label in labels:
            label_counts[label] = train_df_orig[label].sum()
        label_counts = {k: v for k, v in sorted(label_counts.items(), key=lambda item: item[1], reverse=True)}
        top20_labels = list(islice(label_counts, 20))
        other_labels = [x for x in list(label_counts.keys()) if x not in top20_labels]

        train_df_orig.drop(other_labels,inplace=True, axis=1)
        test_df.drop(other_labels,inplace=True, axis=1)

        labels = top20_labels

        np.save(f"{DATA_DIR}/reuters/labels.npy", labels)
        num_labels = len(labels)
        mlflowLogger.store_param("col_names", labels)
        mlflowLogger.store_param("num_labels", num_labels)

        train_df_orig['labels'] = train_df_orig.apply(lambda row: np.array(row[labels].to_list()), axis=1)
        test_df['labels'] = test_df.apply(lambda row: np.array(row[labels].to_list()), axis=1)

        train_df_orig.drop(labels, inplace=True, axis=1)
        test_df.drop(labels, inplace=True, axis=1)

        #-------shuffle
        train_df_orig = train_df_orig.sample(frac=1).reset_index(drop=True)
        mlflowLogger.store_param("dataset.len", len(train_df_orig)+ len(test_df))

        #-------save the datafarme
        train_df_orig_tosave = train_df_orig.copy()
        train_df_orig_tosave['labels'] = train_df_orig_tosave.apply(lambda row: list(row["labels"]), axis=1)
        train_df_orig_tosave.to_csv(f'{DATA_DIR}/reuters/reuters_train.csv', index=False)
        del train_df_orig_tosave

        test_df_tosave = test_df.copy()
        test_df_tosave['labels'] = test_df_tosave.apply(lambda row: list(row["labels"]), axis=1)
        test_df_tosave.to_csv(f'{DATA_DIR}/reuters/reuters_test.csv', index=False)
        del test_df_tosave

        return train_df_orig, test_df, labels, num_labels

def _setup_dataset(dataset_name, dataset_args, tokenizer, tokenizer_args, head_type, head_args, batch_size, model_cfg, training_cv = False, fold = 2): 
    train_df_orig, test_df, labels, num_labels = reuters_dataset_preprocess(dataset_args)

    train_indexes, val_indexes = iterative_train_test_split(train_df_orig['text'], np.array(train_df_orig['labels'].to_list()), 0.15)
    val_df = train_df_orig.iloc[val_indexes,:]
    train_df = train_df_orig.iloc[train_indexes,:]

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    train_dataloader, validation_dataloader, test_dataloader, num_labels = preprocess(train_df, test_df, val_df, tokenizer, tokenizer_args, labels, labels_dict, head_type, head_args, num_labels, model_cfg, batch_size)
    return train_dataloader, validation_dataloader, test_dataloader, num_labels

def _setup_dataset_cv(dataset_name, dataset_args, tokenizer, tokenizer_args, head_type, head_args, batch_size, model_cfg, fold):

    DATA_DIR = dataset_args['root']
    train_df_orig, test_df, labels, num_labels = reuters_dataset_preprocess(dataset_args)
    fold_i =0
    test_df.reset_index(drop=True, inplace=True)

    stratifier = IterativeStratification(n_splits=fold, order=2)
    for train_indexes, val_indexes in stratifier.split(train_df_orig['text'], np.array(train_df_orig['labels'].to_list())):
        fold_i += 1
        print(f"[dataset] ======================================= Fold {fold_i} =======================================")
        if not os.path.exists(f'{DATA_DIR}/{dataset_args["data_path"]}/train_fold{fold_i}.csv'):
            # print("&"*40)
            val_df = train_df_orig.iloc[val_indexes,:]
            train_df = train_df_orig.iloc[train_indexes,:]

            train_df.reset_index(drop=True, inplace=True)
            val_df.reset_index(drop=True, inplace=True)

            save_fold_train_validation(train_df, val_df, dataset_args, fold_i)
        else:
            # print("$"*40)
            train_df, val_df = read_fold_train_validattion(fold_i, dataset_args)

        train_dataloader, validation_dataloader, test_dataloader, num_labels = preprocess_cv(train_df, test_df, val_df, tokenizer, tokenizer_args, labels, labels_dict, head_type, head_args, num_labels, model_cfg, batch_size, fold_i)
        yield train_dataloader, validation_dataloader, test_dataloader, num_labels


labels_dict = {
    'earn': "Earnings and Earnings Forecasts", 
    'acq': "Mergers Acquisitions", 
    'money-fx': "Money Foreign Exchange", 
    'grain': "grain", 
    'crude': "crude (in a natural or raw state, not yet processed or refined.)", 
    'trade': "trade", 
    'interest': "interest", 
    'wheat': "wheat", 
    'ship': "ship", 
    'corn': "corn", 
    'money-supply': "money supply", 
    'dlr': "dollar", 
    'sugar': "sugar", 
    'oilseed': "oilseed", 
    'coffee': "coffee", 
    'gnp': "Gross National Product (total value of a nation's goods and services)", 
    'gold': "gold", 
    'veg-oil': "Vegetable oil", 
    'soybean': "soybean", 
    'bop': "Balance Of Payments",
}

labels_dict_long = {
    'earn': "Earnings and Earnings Forecasts", 
    'acq': "Mergers Acquisitions", 
    'money-fx': "Money Foreign Exchange", 
    'grain': "grain", 
    'crude': "crude (in a natural or raw state, not yet processed or refined.)", 
    'trade': "trade", 
    'interest': "interest", 
    'wheat': "wheat", 
    'ship': "ship", 
    'corn': "corn", 
    'money-supply': "money supply", 
    'dlr': "dollar", 
    'sugar': "sugar", 
    'oilseed': "oilseed", 
    'coffee': "coffee", 
    'gnp': "Gross National Product (total value of a nation's goods and services)", 
    'gold': "gold", 
    'veg-oil': "Vegetable oil", 
    'soybean': "soybean", 
    'bop': "Balance Of Payments (statement of all transactions made between entities in one country and the rest of the world over a defined period of time, such as a quarter or a year.)",
}
