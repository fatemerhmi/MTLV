import os
from pyunpack import Archive
import pandas as pd
import glob
import re 
from xml.dom import minidom
from tqdm import tqdm
import numpy as np
from collections import Counter, OrderedDict
from itertools import islice
from prettytable import PrettyTable
from copy import deepcopy
from skmultilearn.model_selection import IterativeStratification
import torch
import io
import ast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


import logging
from mtl.datasets.utils import download_from_url, extract_archive, unicode_csv_reader
from mtl.heads.utils import padding_heads, group_heads
import mtl.utils.logger as mlflowLogger 

"""
    You can manually donwload the OpenIdataset and put it in the data directory in root. 
    Description of the dataset: https://openi.nlm.nih.gov/faq
    Downlaod from: https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz

    If it's not not already in the data/OpenI directory, we will download it for you
"""

def create_dataLoader(input, labels, batch_size):
    data = TensorDataset(input.input_ids, input.attention_mask, labels)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader

def getfilename(xml_file):
    x = re.search("[ \w-]+?(?=\.)", xml_file)
    return x.group()

def get_report(xml_file):
    #get radiology report text
    data = {}
    mydoc = minidom.parse(xml_file)
    elements = mydoc.getElementsByTagName('AbstractText')
    for element in elements:
        txt = np.nan
        if element.firstChild != None:
            if element.firstChild.data == "None.":
                txt = np.nan
            else:
                txt = element.firstChild.data
        data[element.attributes['Label'].value] = txt
        
    return data

def get_labels(xml_file):
    data = {}
    mydoc = minidom.parse(xml_file)
    elements = mydoc.getElementsByTagName('major')
    txt = []
    for element in elements:
        if element.firstChild != None:
            txt.append(element.firstChild.data)
    if len(txt) == 0:
        txt = np.nan
    data['expert_labels'] = txt
    
    elements = mydoc.getElementsByTagName('automatic')
    txt = []
    for element in elements:
        if element.firstChild != None:
            txt.append(element.firstChild.data)
    if len(txt) == 0:
        txt = np.nan
    data['manual_labels'] = txt
    
    return data

def convert_xml2csv(allFiles):
    columns = ["fileNo", "COMPARISON", "INDICATION", "FINDINGS", "IMPRESSION", "expert_labels", "manual_labels"]
    xml_df = pd.DataFrame(columns = columns)
    for xml_file in tqdm(allFiles):
        row = {}
        row['fileNo'] = getfilename(xml_file)

        reportDic = get_report(xml_file)
        labelsDic = get_labels(xml_file)

        row = {**row ,**reportDic, **labelsDic}
        xml_df = xml_df.append(row,ignore_index=True)
    return xml_df

def plot_Barchart_top_n_labels(n, openI_df):
    expert_labels = []
    for labels in openI_df['expert_labels']:
        for label in labels:
            expert_labels.append(label)

    label_counts = Counter(expert_labels)

    sorted_label_counts = OrderedDict(sorted(label_counts.items(), key=lambda x: x[1], reverse=True))
    
    unique_labels = sorted_label_counts.keys()
    # print("Total No. of Unique labels:",len(unique_labels))
    
    sliced = islice(sorted_label_counts.items(), n)  
    sliced_o = OrderedDict(sliced)

    df = pd.DataFrame.from_dict(sliced_o, orient='index')
#     df.plot(kind='bar', title = "Top "+str(n)+" ranked expert labels")
    return unique_labels
def find_similar_disorders_caseInsensitive(disorder,unique_labels):
    similar_disorders = []
    for label in unique_labels:
        match = re.search(".*"+disorder+".*", label, flags = re.IGNORECASE)
        if match is not None:
            similar_disorders.append(match.group())
#             print(match.group())
    return similar_disorders

def find_similar_disorders_caseSensitive(disorder,unique_labels):
    similar_disorders = []
    for label in unique_labels:
        match = re.search(".*"+disorder+".*", label)
        if match is not None:
            similar_disorders.append(match.group())
#             print(match.group())
    return similar_disorders

def update_row(row,similar_disorders,disorder):
#     print("Before:", row)
    new_label_list = []
    for item in row:
        if item in similar_disorders:
            new_label_list.append(disorder)
        else:
            new_label_list.append(item)
#     print("After:",new_label_list)
    return new_label_list

def update_labels(similar_disorders, disorder, openI_df):
    # print("Updating similar labels to :", disorder)
    openI_df['expert_labels'] = openI_df.apply(lambda row: \
                                               update_row(row['expert_labels'],similar_disorders,disorder), \
                                               axis=1)
    return openI_df
    
def create_new_column(column_name, openI_df):
    openI_df[column_name] = openI_df.apply(lambda row: \
                                           1 if column_name in row['expert_labels'] \
                                           else 0, \
                                           axis=1)
    return openI_df

def iterative_train_test_split(X, y, test_size):
    stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[test_size, 1.0-test_size])
    train_indexes, test_indexes = next(stratifier.split(X, y))

    return train_indexes, test_indexes

def concat_cols(impression, findings):
    if impression is np.nan:
        return findings
    elif findings is np.nan:
        return impression
    else:
        return findings+impression

def _setup_datasets(dataset_name, dataset_args, tokenizer, tokenizer_args, head_type, head_args, batch_size): # previously read_data
    #------- download and set up openI dataset
    DATA_DIR = dataset_args['root']

    if os.path.exists(f"{DATA_DIR}/OpenI"):
        print("[  dataset  ] OpenI directory already exists")
        openI_df = pd.read_csv(f"{DATA_DIR}/{dataset_args['data_path']}")
        openI_df['expert_labels'] = openI_df.apply(lambda row: ast.literal_eval(row['expert_labels']), axis=1)
    else:
        os.makedirs(f"{DATA_DIR}/OpenI")
        
        print("[  dataset  ] openI dataset is being downloaded...")
        os.system(f'wget -N -P {DATA_DIR}/OpenI https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz')
        
        print("[  dataset  ] Extracting openI dataset...")
        directory_to_extract_to = f"{DATA_DIR}/OpenI/"
        Archive(f"{DATA_DIR}/OpenI/NLMCXR_reports.tgz").extractall(directory_to_extract_to)
        os.system(f"rm {DATA_DIR}/OpenI/NLMCXR_reports.tgz")

        #-------Convert XML to csv
        allFiles = glob.glob(f'{DATA_DIR}/OpenI/ecgen-radiology/**/*.xml', recursive=True)
        openI_df = convert_xml2csv(allFiles)
        openI_df = openI_df.drop(['manual_labels'], axis=1)
        os.system(f"rm -r {DATA_DIR}/OpenI/ecgen-radiology") 
        openI_df.to_csv(f'{DATA_DIR}/OpenI/openI.csv', index=False)
    
    #-------fine the labels acording to the previous citations-----------
    paper1 = ['Atelectasis', 'Cardiomegaly', 'Effusion', 
        'Infiltrate', 'Mass', 'Nodule', 
        'Pneumonia', 'Pneumothorax']

    paper2 = ['Pneumonia', 'Emphysema', 'Effusion', 'Consolidation', 
            'Nodule', 'Atelectasis', 'Edema', 'Cardiomegaly', 'Hernia']

    paper3 = ['Atelectasis', 'Cardiomegaly', 'Effusion', 
            'Infiltrate', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis']

    paper4 = ['opacity', 'Cardiomegaly', 'Calcinosis', 'lung/hypoinflation',
            'Calcified granuloma', 'thoracic vertebrae/degenerative', 'lung/hyperdistention', 
            'spine/degenerative', 'catheters, indwelling', 'granulomatous disease', 
            'nodule', 'surgical instrument']

    added = ['Cicatrix', 'Deformity', "Medical Device", "Airspace Disease"]

    labels = list(set().union(paper1, paper2, paper3, paper4, added))
    labels.sort()

    unique_labels = plot_Barchart_top_n_labels(20, openI_df)
    for label in labels: 
        disorder = label
        if disorder == "Pneumothorax":
            similar_disorders = find_similar_disorders_caseSensitive(disorder, unique_labels)
        else:
            similar_disorders = find_similar_disorders_caseInsensitive(disorder, unique_labels)
        if similar_disorders != []:
            openI_df = update_labels(similar_disorders, disorder, openI_df)
            openI_df =  create_new_column(label, openI_df)
            unique_labels = plot_Barchart_top_n_labels(20, openI_df)
        else:
            print(label)

    #-------preprocess finding and impression
    print('No. of rows with Finding:', len(openI_df[openI_df['FINDINGS'].notnull()]))
    print('No. of rows with Impression:', len(openI_df[openI_df['IMPRESSION'].notnull()]))
    print('No. of rows with Impression or Finding:', 
        len(openI_df[openI_df['IMPRESSION'].notnull() | openI_df['FINDINGS'].notnull()]))
    print('No. of rows without Impression and Finding:', 
        len(openI_df[openI_df['IMPRESSION'].isna() & openI_df['FINDINGS'].isna()]))

    idx = openI_df[openI_df['IMPRESSION'].isna() & openI_df['FINDINGS'].isna()].index
    openI_df.drop(idx, inplace = True)
    print('No. of rows without Impression and Finding:', 
        len(openI_df[openI_df['IMPRESSION'].isna() & openI_df['FINDINGS'].isna()]))

    #-------drop unnecessary columns
    to_drop = []
    for label in labels:
        if openI_df[label].sum()<100:
            to_drop.append(label)

    openI_df.drop(['fileNo','expert_labels','COMPARISON','INDICATION' ]+ to_drop, inplace=True, axis=1)

    #-------rename columns
    openI_df.rename(columns={"lung/hyperdistention": "lung hyperdistention", \
                   "lung/hypoinflation": "lung hypoinflation", \
                    "spine/degenerative": "spine degenerative", \
                    "thoracic vertebrae/degenerative": "thoracic vertebrae degenerative", \
                    "catheters, indwelling":"indwelling catheters",
                   }, inplace= True)

    #-------log how many labels + label cols
    cols = openI_df.columns
    labels = list(cols[2:])
    num_labels = len(labels)
    mlflowLogger.store_param("col_names", labels)
    mlflowLogger.store_param("num_labels", num_labels)
    # print('Count of 1 per label: \n', openI_df[labels].sum(), '\n') # Label counts, may need to downsample or upsample

    #-------Converting the labels to a single list
    df_cls = pd.DataFrame(columns = ['text', 'labels'])

    #-------create the text column:
    df_cls['text'] = openI_df.apply(lambda row: concat_cols(row['IMPRESSION'], row['FINDINGS']), axis=1)
    df_cls['labels'] = openI_df.apply(lambda row: np.array(row[labels].to_list()), axis=1) #.to_list() , .values

    print('Unique texts: ', df_cls.text.nunique() == df_cls.shape[0])

    print("Length of whole dataframe:", len(df_cls))
    print("No. of unique reports:", df_cls.text.nunique())

    #-------remove the duplicates
    df_cls.drop_duplicates('text', inplace=True)

    print('Unique texts: ', df_cls.text.nunique() == df_cls.shape[0])
    #-------shuffle
    df_cls = df_cls.sample(frac=1).reset_index(drop=True)

    #-------stratified sampling
    train_indexes, test_indexes = iterative_train_test_split(df_cls['text'], np.array(df_cls['labels'].to_list()), 0.2)

    train_df = df_cls.iloc[train_indexes,:]
    test_df = df_cls.iloc[test_indexes,:]

    train_indexes, val_indexes = iterative_train_test_split(train_df['text'], np.array(train_df['labels'].to_list()), 0.15)

    train_df = df_cls.iloc[train_indexes,:]
    val_df = df_cls.iloc[val_indexes,:]

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    print('Train: ', len(train_df))
    print('Test: ', len(test_df))
    print('Val: ', len(val_df))

    #-------table for tran, test,val counts
    label_counts_total = np.array(df_cls.labels.to_list()).sum(axis=0)
    label_counts_train = np.array(train_df.labels.to_list()).sum(axis=0)
    label_counts_test = np.array(test_df.labels.to_list()).sum(axis=0)
    label_counts_val = np.array(val_df.labels.to_list()).sum(axis=0)


    pretty=PrettyTable()
    pretty.field_names = ['Pathology', 'total', 'train', 'test','val']
    for pathology, cnt_total, cnt_train, cnt_test, cnt_val in zip(labels, label_counts_total, label_counts_train, label_counts_test, label_counts_val):
        pretty.add_row([pathology, cnt_total, cnt_train, cnt_test, cnt_val])
    print(pretty)

    #-------read the data
    # data_path =  f"{dataset_args['root']}/{dataset_args['data_path']}"        
    # split_path = f"{dataset_args['root']}/{dataset_args['split_path']}"  

    # df = pd.read_csv(data_path)
    # cols = df.columns
    # label_cols = list(cols[6:])
    # num_labels = len(label_cols)

    #-------log how many labels + label cols
    # mlflowLogger.store_param("col_names", label_cols)
    # mlflowLogger.store_param("num_labels", num_labels)

    #-------convert string of list to actual list (labels)
    # train_df = pd.read_csv(f"{split_path}/train.csv")
    # train_df['labels'] = train_df.apply(lambda row: ast.literal_eval(row['labels']), axis=1)

    # test_df = pd.read_csv(f"{split_path}/test.csv")
    # test_df['labels'] = test_df.apply(lambda row: ast.literal_eval(row['labels']), axis=1)

    # val_df = pd.read_csv(f"{split_path}/val.csv")
    # val_df['labels'] = val_df.apply(lambda row: ast.literal_eval(row['labels']), axis=1)
    
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