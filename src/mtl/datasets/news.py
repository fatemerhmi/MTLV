import ast
import pandas as pd
from skmultilearn.model_selection import IterativeStratification
import numpy as np
import os 
import re

from mtl.datasets.utils import download_from_url, extract_archive, unicode_csv_reader, preprocess, preprocess_cv
from mtl.heads.utils import padding_heads, group_heads
import mtl.utils.logger as mlflowLogger 
from mtl.datasets.utils import iterative_train_test_split, create_dataLoader
from mtl.heads.grouping_KDE import *
from mtl.heads.grouping_meanshift import *
from mtl.heads.grouping_kmediod import grouping_kmediod, get_all_label_embds, plot_elbow_method
import mtl.utils.configuration as configuration

def prepare_text_col(df_train, df_test):
    df_train.replace(np.nan, "", inplace=True)
    df_test.replace(np.nan, "", inplace=True)

    df_train.loc[:,'labels'] = df_train.apply(lambda row: ast.literal_eval(row['labels']), axis=1)
    df_test.loc[:,'labels'] = df_test.apply(lambda row: ast.literal_eval(row['labels']), axis=1)

    df_train.loc[:,'text'] = df_train.apply(lambda row: row['title']+" "+ row['desc'], axis=1)
    df_test.loc[:,'text'] = df_test.apply(lambda row: row['title']+" "+ row['desc'], axis=1)

    # remove columns except title and labels
    columns_to_remove = list(df_train.columns)
    columns_to_remove.remove("text") 
    columns_to_remove.remove("labels")
    # print(columns_to_remove)
    df_train.drop(columns= columns_to_remove, inplace=True)
    df_test.drop(columns= columns_to_remove, inplace=True)
    return df_train, df_test

def news_dataset_preprocess(dataset_args):
    #------- download and set up openI dataset
    DATA_DIR = dataset_args['root']

    if os.path.exists(f"{DATA_DIR}/{dataset_args['data_path']}"):
        #---------load dataframe
        print("[  dataset  ] news directory already exists")
       
        #--------load dataframe
        train_df_orig = pd.read_csv(f"{DATA_DIR}/news/train.csv")
        test_df = pd.read_csv(f"{DATA_DIR}/news/test.csv")

        mlflowLogger.store_param("dataset.len", len(train_df_orig)+len(test_df))

        #--------loading and storing labels to mlflow
        cols = train_df_orig.columns
        labels = list(cols[3:-1])
        labels = [re.sub(r'_+', ' ', s) for s in labels]
        num_labels = len(labels)
        mlflowLogger.store_param("col_names", labels)
        mlflowLogger.store_param("num_labels", num_labels)

        #---------prepare the dataset
        train_df_orig, test_df = prepare_text_col(train_df_orig, test_df)

        return train_df_orig, test_df, labels, num_labels
    else:
        raise Exception(f"{DATA_DIR}/{dataset_args['data_path']} does not exists!")

def _setup_dataset(dataset_name, dataset_args, tokenizer, tokenizer_args, head_type, head_args, batch_size, model_cfg): 
    
    train_df_orig, test_df, labels, num_labels = news_dataset_preprocess(dataset_args)

    train_indexes, val_indexes = iterative_train_test_split(train_df_orig['text'], np.array(train_df_orig['labels'].to_list()), 0.15)
    val_df = train_df_orig.iloc[val_indexes,:]
    train_df = train_df_orig.iloc[train_indexes,:]

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    train_dataloader, validation_dataloader, test_dataloader, num_labels = preprocess(train_df, test_df, val_df, tokenizer, tokenizer_args, labels, labels_dict, head_type, head_args, num_labels, model_cfg, batch_size)
    return train_dataloader, validation_dataloader, test_dataloader, num_labels

def _setup_dataset_cv(dataset_name, dataset_args, tokenizer, tokenizer_args, head_type, head_args, batch_size, model_cfg, fold):

    train_df_orig, test_df, labels, num_labels = news_dataset_preprocess(dataset_args)

    fold_i =0
    stratifier = IterativeStratification(n_splits=fold, order=2)
    for train_indexes, val_indexes in stratifier.split(train_df_orig['text'], np.array(train_df_orig['labels'].to_list())):
        fold_i += 1
        print(f"[dataset] ======================================= Fold {fold_i} =======================================")

        val_df = train_df_orig.iloc[val_indexes,:]
        train_df = train_df_orig.iloc[train_indexes,:]

        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)

        train_dataloader, validation_dataloader, test_dataloader, num_labels = preprocess_cv(train_df, test_df, val_df, tokenizer, tokenizer_args, labels, labels_dict, head_type, head_args, num_labels, model_cfg, batch_size, fold_i)
        yield train_dataloader, validation_dataloader, test_dataloader, num_labels

def _setup_dataset_ttest(dataset_name, dataset_args, tokenizer, tokenizer_args, head_type, head_args, batch_size, model_cfg, fold):

    train_df_orig, test_df, labels, num_labels = news_dataset_preprocess(dataset_args)

    fold_i =0
    stratifier = IterativeStratification(n_splits=fold, order=2)
    for train_indexes, val_indexes in stratifier.split(train_df_orig['text'], np.array(train_df_orig['labels'].to_list())):
        fold_i += 1
        print(f"[dataset] ======================================= Fold {fold_i} =======================================")

        val_df = train_df_orig.iloc[val_indexes,:]
        train_df = train_df_orig.iloc[train_indexes,:]

        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)

        train_dataloader_s, validation_dataloader_s, test_dataloader_s, num_labels_s = preprocess_cv(train_df, test_df, val_df, tokenizer, tokenizer_args, labels, labels_dict, "single-head", head_args, num_labels, model_cfg, batch_size, fold_i)
        train_dataloader_mtl, validation_dataloader_mtl, test_dataloader_mtl, num_labels_mtl = preprocess_cv(train_df, test_df, val_df, tokenizer, tokenizer_args, labels, labels_dict, "multi-task", head_args, num_labels, model_cfg, batch_size, fold_i)
        yield train_dataloader_s, validation_dataloader_s, test_dataloader_s, num_labels_s, train_dataloader_mtl, validation_dataloader_mtl, test_dataloader_mtl, num_labels_mtl

labels_dict={
    'new service launch': "",
    'executive change': "",
    'award and recognition': "",
    'new product launch': "",
    'divestment': "",
    'financial loss': "",
    'customer loss growth decline': "",
    'organizational restructuring': "",
    'merger acquisition': "",
    'cost cutting': "",
    'business shut-down': "",
    'bankruptcy': "",
    'regulatory settlement': "",
    'regulatory investigation': "",
    'initial public offering': "",
    'joint venture': "",
    'hiring': "",
    'financial result': "",
    'fundraising investment': "",
    'executive appointment': "",
    'regulatory approval': "",
    'downsizing': "",
    'product shutdown': "",
    'pricing': "",
    'new fund launch': "",
    'going private': "",
    'investment exit': "",
    'employee dispute strike': "",
    'delist': "",
    'business outlook projections': "",
    'company executive statement': "",
    'board decisions': "",
    'business expansion': "",
    'buyout': "",
    'alliance partnership': "",
    'law suit judgement settlement': ""
}