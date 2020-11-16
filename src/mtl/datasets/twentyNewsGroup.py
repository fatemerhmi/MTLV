from skmultilearn.model_selection import IterativeStratification
from prettytable import PrettyTable
import os
from pyunpack import Archive
import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
import torch

from mtl.datasets.utils import iterative_train_test_split, create_dataLoader, create_new_column, preprocess, preprocess_cv
import mtl.utils.logger as mlflowLogger 
 
def make_csv(path, labels):
    all_data = []
    for label in tqdm(labels):
        instances_in_a_label = os.listdir (f"{path}/{label}")
        for item in instances_in_a_label:
            f = open(f"{path}/{label}/{item}", "r" , encoding='utf-8',errors='ignore' )
            raw_data = f.read()
            all_data.append([item, raw_data, label])
    all_data = np.asarray(all_data)
    df = pd.DataFrame(all_data, columns=["id", "text", "label"])
    return df

def twentyNewsGroup_preprocess(dataset_args):
    #------- download and set up openI dataset
    DATA_DIR = dataset_args['root']

    if os.path.exists(f"{DATA_DIR}/twentynewsgroup"):
        #--------load dataframe
        print("[  dataset  ] twentynewsgroup directory already exists.")
        train_df_orig = pd.read_csv(f"{DATA_DIR}/twentynewsgroup/twentynewsgroup_train.csv")
        train_df_orig['labels'] = train_df_orig.apply(lambda row: np.array(ast.literal_eval(row['labels'])), axis=1)

        test_df = pd.read_csv(f"{DATA_DIR}/twentynewsgroup/twentynewsgroup_test.csv")
        test_df['labels'] = test_df.apply(lambda row: np.array(ast.literal_eval(row['labels'])), axis=1)

        # #--------loading and storing labels to mlflow
        labels = np.load(f"{DATA_DIR}/twentynewsgroup/labels.npy")
        labels = list(labels)
        num_labels = len(labels)
        mlflowLogger.store_param("col_names", labels)
        mlflowLogger.store_param("num_labels", num_labels)

        train_df_orig, test_df, labels, num_labels

    else: 
        os.makedirs(f"{DATA_DIR}/twentynewsgroup")

        print("[  dataset  ] twentynewsgroup dataset is being downloaded...")
        os.system(f'wget -N -P {DATA_DIR}/twentynewsgroup http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz')

        # data/twentynewsgroup/20news-bydate.tar.gz
        print("[  dataset  ] Extracting twentynewsgroup dataset...")
        directory_to_extract_to = f"{DATA_DIR}/twentynewsgroup/"
        Archive(f"{DATA_DIR}/twentynewsgroup/20news-bydate.tar.gz").extractall(directory_to_extract_to)
        os.system(f"rm {DATA_DIR}/twentynewsgroup/20news-bydate.tar.gz")

        #------storing label details to mlflow and npy file
        labels = os.listdir (f"{DATA_DIR}/twentynewsgroup/20news-bydate-train")
        if ".DS_Store" in labels:
            labels.remove(".DS_Store")

        np.save(f"{DATA_DIR}/twentynewsgroup/labels.npy", labels)
        num_labels = len(labels)
        mlflowLogger.store_param("col_names", labels)
        mlflowLogger.store_param("num_labels", num_labels)

        #------convert the files to a dataframe 
        train_df_orig = make_csv(f"{DATA_DIR}/twentynewsgroup/20news-bydate-train", labels)
        test_df = make_csv(f"{DATA_DIR}/twentynewsgroup/20news-bydate-test", labels)

        os.system(f"rm -r {DATA_DIR}/twentynewsgroup/20news-bydate-*")

            #------preprocessing the labels
        tqdm.pandas()
        print("\n[  dataset  ] twentynewsgroup preprocessing of labels begin...")
        train_df_orig["labels"] = train_df_orig.progress_apply(lambda row: train_df_orig.loc[train_df_orig['id'] == row['id']].label.tolist(), axis=1)
        test_df["labels"] = test_df.progress_apply(lambda row: test_df.loc[test_df['id'] == row['id']].label.tolist(), axis=1)

        os.system(f"rm -r {DATA_DIR}/twentynewsgroup/20news-bydate-*")

        #------remove duplicate rows
        train_df_orig.drop_duplicates('id', inplace=True)
        test_df.drop_duplicates('id', inplace=True)

        #------bring labels to seperate columns
        for label in tqdm(labels):
            train_df_orig = create_new_column(train_df_orig, label)
            test_df = create_new_column(test_df, label)

        train_df_orig.drop(['labels', "label"],inplace=True, axis=1)
        test_df.drop(['labels', "label"],inplace=True, axis=1)

        train_df_orig['labels'] = train_df_orig.apply(lambda row: np.array(row[labels].to_list()), axis=1)
        test_df['labels'] = test_df.apply(lambda row: np.array(row[labels].to_list()), axis=1)

        train_df_orig.drop(labels, inplace=True, axis=1)
        test_df.drop(labels, inplace=True, axis=1)

        #------remove back quote from text
        train_df_orig['text'] = train_df_orig.apply(lambda row: row.text.replace("`", "'"), axis=1)
        test_df['text'] = test_df.apply(lambda row: row.text.replace("`", "'"), axis=1)

        #-------save the datafarme
        # train_df_orig.to_csv(f'{DATA_DIR}/twentynewsgroup/twentynewsgroup_train.csv', index=False)
        # test_df.to_csv(f'{DATA_DIR}/twentynewsgroup/twentynewsgroup_test.csv', index=False)

        #-------save the datafarme
        train_df_orig_tosave = train_df_orig.copy()
        train_df_orig_tosave['labels'] = train_df_orig_tosave.apply(lambda row: list(row["labels"]), axis=1)
        train_df_orig_tosave.to_csv(f'{DATA_DIR}/twentynewsgroup/twentynewsgroup_train.csv', index=False)
        
        test_df_tosave = test_df.copy()
        test_df_tosave['labels'] = test_df_tosave.apply(lambda row: list(row["labels"]), axis=1)
        test_df_tosave.to_csv(f'{DATA_DIR}/twentynewsgroup/twentynewsgroup_test.csv', index=False)
        
        train_df_orig, test_df, labels, num_labels

def _setup_dataset(dataset_name, dataset_args, tokenizer, tokenizer_args, head_type, head_args, batch_size, model_cfg, training_cv = False, fold = 2): # previously read_data
    
    train_df_orig, test_df, labels, num_labels = twentyNewsGroup_preprocess(dataset_args)

    train_indexes, val_indexes = iterative_train_test_split(train_df_orig['text'], np.array(train_df_orig['labels'].to_list()), 0.15)
    val_df = train_df_orig.iloc[val_indexes,:]
    train_df = train_df_orig.iloc[train_indexes,:]

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    train_dataloader, validation_dataloader, test_dataloader, num_labels = preprocess(train_df, test_df, val_df, tokenizer, tokenizer_args, labels, labels_dict, head_type, head_args, num_labels, model_cfg, batch_size)
    return train_dataloader, validation_dataloader, test_dataloader, num_labels

def _setup_dataset_cv(dataset_name, dataset_args, tokenizer, tokenizer_args, head_type, head_args, batch_size, model_cfg, fold):

    train_df_orig, test_df, labels, num_labels = twentyNewsGroup_preprocess(dataset_args)

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

    # #-------shuffle
    # train_val_df = train_val_df.sample(frac=1).reset_index(drop=True)
    # mlflowLogger.store_param("dataset.len", len(train_val_df)+ len(test_df))

    # #-------stratified sampling
    # train_indexes, val_indexes = iterative_train_test_split(train_val_df['text'], np.array(train_val_df['labels'].to_list()), 0.15)

    # train_df = train_val_df.iloc[train_indexes,:]
    # val_df = train_val_df.iloc[val_indexes,:]

    # train_df.reset_index(drop=True, inplace=True)
    # test_df.reset_index(drop=True, inplace=True)
    # val_df.reset_index(drop=True, inplace=True)

    # print('Train: ', len(train_df))
    # mlflowLogger.store_param("dataset.train.len", len(train_df))
    # print('Test: ', len(test_df))
    # mlflowLogger.store_param("dataset.test.len", len(test_df))
    # print('Val: ', len(val_df))
    # mlflowLogger.store_param("dataset.val.len", len(val_df))

    # #-------table for train, test, val counts
    # label_counts_train = np.array(train_df.labels.to_list()).sum(axis=0)
    # label_counts_test = np.array(test_df.labels.to_list()).sum(axis=0)
    # label_counts_val = np.array(val_df.labels.to_list()).sum(axis=0)


    # pretty=PrettyTable()
    # label_counts_total = []
    # pretty.field_names = ['Label', 'total', 'train', 'test','val']
    # for pathology, cnt_train, cnt_test, cnt_val in zip(labels, label_counts_train, label_counts_test, label_counts_val):
    #     cnt_total = cnt_train + cnt_test + cnt_val
    #     label_counts_total.append(cnt_total)
    #     pretty.add_row([pathology, cnt_total, cnt_train, cnt_test, cnt_val])
    # print(pretty)

    #  #-------check for multi-head, single or multi-task
    # if head_type =="multi-task":
    #     #check the type:   label_counts_total
    #     if head_args['type'] == "givenset":
    #         heads_index = head_args["heads_index"]

    #     elif head_args['type'] == "KDE":
    #         print("[  dataset  ] KDE label grouping starts!")
    #         heads_index = KDE(label_counts_total, head_args['bandwidth'])

    #     elif head_args['type'] == "meanshift":
    #         print("[  dataset  ] meanshift label grouping starts!")
    #         heads_index = meanshift(label_counts_total)

    #     elif head_args['type'] == "kmediod-label":
    #         print("[  dataset  ] kmediod-label grouping starts!")
    #         model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
    #         embds = get_all_label_embds(labels, tokenizer, model)
    #         if "elbow" in head_args.keys():
    #             plot_elbow_method(embds,head_args['elbow'])
    #         heads_index = grouping_kmediod(embds, head_args['clusters'])
    #         del model

    #     elif head_args['type'] == "kmediod-labeldesc":
    #         print("[  dataset  ] kmediod-label description grouping starts!")
    #         model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
    #         labels_list = [labels_dict[label] for label in labels]
    #         # list(labels_dict.values()
    #         embds = get_all_label_embds(labels_list, tokenizer, model)
    #         if "elbow" in head_args.keys():
    #             plot_elbow_method(embds,head_args['elbow'])
    #         heads_index = grouping_kmediod(embds, head_args['clusters'])
    #         del model

    #     mlflowLogger.store_param("heads_index", heads_index)
    #     padded_heads = padding_heads(heads_index)
        
    #     #--group the heads
    #     train_df = group_heads(padded_heads, train_df)
    #     test_df = group_heads(padded_heads, test_df)
    #     val_df = group_heads(padded_heads, val_df)

    #     #--prepare labels for dataloader
    #     train_labels = torch.from_numpy(np.array(train_df.head_labels.to_list()))
    #     test_labels = torch.from_numpy(np.array(test_df.head_labels.to_list()))
    #     val_labels = torch.from_numpy(np.array(val_df.head_labels.to_list()))

    # elif head_type =="single-head":

    #     #--prepare labels for dataloader
    #     train_labels = torch.from_numpy(np.array(train_df.labels.to_list()))
    #     test_labels = torch.from_numpy(np.array(test_df.labels.to_list()))
    #     val_labels = torch.from_numpy(np.array(val_df.labels.to_list()))
    # else:
    #     raise Exception("The head type must be either 'multi-task' or 'single-head'!")

    # #-------tokenize
    # reports_train = train_df.text.to_list()
    # reports_test = test_df.text.to_list()
    # reports_val   = val_df.text.to_list()
   
    # train = tokenizer(reports_train, padding=tokenizer_args['padding'], truncation=tokenizer_args['truncation'], max_length=tokenizer_args['max_length'], return_tensors="pt")
    # test = tokenizer(reports_test, padding=tokenizer_args['padding'], truncation=tokenizer_args['truncation'], max_length=tokenizer_args['max_length'], return_tensors="pt")
    # val = tokenizer(reports_val, padding=tokenizer_args['padding'], truncation=tokenizer_args['truncation'], max_length=tokenizer_args['max_length'], return_tensors="pt")
    
    # #-------create dataloarders
    # train_dataloader      = create_dataLoader(train, train_labels, batch_size)
    # validation_dataloader = create_dataLoader(val, val_labels, batch_size)
    # test_dataloader       = create_dataLoader(test, test_labels, batch_size)

    # return train_dataloader, validation_dataloader, test_dataloader, num_labels

labels_dict = {
    'comp.windows.x': "computer windows x", 
    'comp.os.ms-windows.misc': "computer os ms-windows misc", 
    'comp.sys.mac.hardware': "computer sys mac hardware", 
    'comp.graphics': "computer graphics", 
    'comp.sys.ibm.pc.hardware': "computer sys ibm pc hardware",
    'talk.politics.guns': "talk politics guns", 
    'soc.religion.christian': "soc religion christian", 
    'talk.religion.misc': "talk religion misc", 
    'talk.politics.misc': "talk politics misc", 
    'talk.politics.mideast': "talk politics mideast", 
    'sci.med': "sci med", 
    'sci.crypt': "science crypt", 
    'sci.space': "science space", 
    'sci.electronics': "science electronics", 
    'rec.autos': "rec autos", 
    'rec.sport.hockey': "rec sport hockey", 
    'rec.sport.baseball': "rec sport baseball", 
    'rec.motorcycles': "rec motorcycles", 
    'misc.forsale': "misc forsale", 
    'alt.atheism': "alt atheism", 
}