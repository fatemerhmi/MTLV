from skmultilearn.model_selection import IterativeStratification
from prettytable import PrettyTable
import os
from pyunpack import Archive
import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
import torch

from mtl.datasets.utils import iterative_train_test_split, create_dataLoader
import mtl.utils.logger as mlflowLogger 

def create_new_column(df, column_name):
    df[column_name] = df.apply(lambda row: \
                                        1 if column_name in row['labels'] \
                                        else 0, \
                                        axis=1)
    return df

def _setup_datasets(dataset_name, dataset_args, tokenizer, tokenizer_args, head_type, head_args, batch_size):
    #------- load or download then set up ohsumed dataset
    DATA_DIR = dataset_args['root']

    if os.path.exists(f"{DATA_DIR}/{dataset_args['data_path']}"):
        #--------load dataframe
        print("[  dataset  ] ohsumed directory already exists.")
        df = pd.read_csv(f"{DATA_DIR}/ohsumed")
        df['labels'] = df.apply(lambda row: ast.literal_eval(row['labels']), axis=1)

        #--------loading and storing labels to mlflow
        labels = np.load("{DATA_DIR}/ohsumed/labels.npy")
        num_labels = len(labels)
        mlflowLogger.store_param("col_names", labels)
        mlflowLogger.store_param("num_labels", num_labels)

    else:
        os.makedirs(f"{DATA_DIR}/ohsumed")
        
        print("[  dataset  ] ohsumed dataset is being downloaded...")
        os.system(f'wget -N -P {DATA_DIR}/ohsumed http://disi.unitn.eu/moschitti/corpora/ohsumed-all-docs.tar.gz')
        os.system(f'wget -N -P {DATA_DIR}/ohsumed http://disi.unitn.eu/moschitti/corpora/First-Level-Categories-of-Cardiovascular-Disease.txt')
        
        print("[  dataset  ] Extracting openI dataset...")
        directory_to_extract_to = f"{DATA_DIR}/ohsumed/"
        Archive(f"{DATA_DIR}/ohsumed/ohsumed-all-docs.tar.gz").extractall(directory_to_extract_to)
        os.system(f"rm {DATA_DIR}/ohsumed/ohsumed-all-docs.tar.gz")

        #------storing label details to mlflow and npy file
        labels = os.listdir (f"{DATA_DIR}/ohsumed/ohsumed-all")
        np.save(f"{DATA_DIR}/ohsumed/labels.npy", labels)
        num_labels = len(labels)
        mlflowLogger.store_param("col_names", labels)
        mlflowLogger.store_param("num_labels", num_labels)

        #------convert the files to a dataframe 
        if ".DS_Store" in labels:
            labels.remove(".DS_Store")
        all_data = []
        for label in tqdm(labels):
            instances_in_a_label = os.listdir (f"{DATA_DIR}/ohsumed/ohsumed-all/{label}")
            for item in instances_in_a_label:
                f = open(f"{DATA_DIR}/ohsumed/ohsumed-all/{label}/{item}", "r")
                raw_data = f.read()
                all_data.append([item, raw_data, label])
        all_data = np.asarray(all_data)
        df = pd.DataFrame(all_data, columns=["id", "text", "label"])

        os.system(f"rm -r {DATA_DIR}/ohsumed/ohsumed-all")

        #------preprocessing the labels
        tqdm.pandas()
        print("\n[  dataset  ] ohsumed preprocessing of labels begin...")
        df["labels"] = df.progress_apply(lambda row: df.loc[df['id'] == row['id']].label.tolist(), axis=1)

        #------remove duplicate rows
        df.drop_duplicates('id', inplace=True)

        #------bring labels to seperate columns
        for label in tqdm(labels):
            df = create_new_column(df, label)

        df.drop(['labels', "label"],inplace=True, axis=1)

        df['labels'] = df.apply(lambda row: np.array(row[labels].to_list()), axis=1)

        df.drop(labels, inplace=True, axis=1)

        #-------save the datafarme
        df.to_csv(f'{DATA_DIR}/ohsumed/ohsumed.csv', index=False)

    #-------shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    mlflowLogger.store_param("dataset.len", len(df))

    #-------stratified sampling
    train_indexes, test_indexes = iterative_train_test_split(df['text'], np.array(df['labels'].to_list()), 0.2)

    train_df = df.iloc[train_indexes,:]
    test_df = df.iloc[test_indexes,:]

    train_indexes, val_indexes = iterative_train_test_split(train_df['text'], np.array(train_df['labels'].to_list()), 0.15)

    train_df = df.iloc[train_indexes,:]
    val_df = df.iloc[val_indexes,:]

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    print('Train: ', len(train_df))
    mlflowLogger.store_param("dataset.train.len", len(train_df))
    print('Test: ', len(test_df))
    mlflowLogger.store_param("dataset.test.len", len(test_df))
    print('Val: ', len(val_df))
    mlflowLogger.store_param("dataset.val.len", len(val_df))

    #-------table for tran, test,val counts
    label_counts_total = np.array(df.labels.to_list()).sum(axis=0)
    label_counts_train = np.array(train_df.labels.to_list()).sum(axis=0)
    label_counts_test = np.array(test_df.labels.to_list()).sum(axis=0)
    label_counts_val = np.array(val_df.labels.to_list()).sum(axis=0)


    pretty=PrettyTable()
    pretty.field_names = ['Label', 'total', 'train', 'test','val']
    for pathology, cnt_total, cnt_train, cnt_test, cnt_val in zip(labels, label_counts_total, label_counts_train, label_counts_test, label_counts_val):
        pretty.add_row([pathology, cnt_total, cnt_train, cnt_test, cnt_val])
    print(pretty)

    #-------check for multi-head, single or multi-task
    if head_type =="multi-task":
        heads_index = head_args["heads_index"]
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

    return train_dataloader, validation_dataloader, test_dataloader