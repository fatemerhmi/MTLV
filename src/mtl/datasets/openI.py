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
import torch
import io
import ast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from skmultilearn.model_selection import IterativeStratification

from mtl.datasets.utils import preprocess, preprocess_cv, save_fold_train_validation, read_fold_train_validattion
from mtl.heads.utils import padding_heads, group_heads
import mtl.utils.logger as mlflowLogger 
from mtl.datasets.utils import iterative_train_test_split, create_dataLoader
import mtl.utils.configuration as configuration


"""
    You can manually donwload the OpenIdataset and put it in the data directory in root. 
    Description of the dataset: https://openi.nlm.nih.gov/faq
    Downlaod from: https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz

    If it's not not already in the data/OpenI directory, we will download it for you
"""

def getfilename(xml_file):
    x = re.search(r"[ \w-]+?(?=\.)", xml_file)
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
    # df.plot(kind='bar', title = "Top "+str(n)+" ranked expert labels")
    return unique_labels

def find_similar_disorders_caseInsensitive(disorder,unique_labels):
    similar_disorders = []
    for label in unique_labels:
        match = re.search(".*"+disorder+".*", label, flags = re.IGNORECASE)
        if match is not None:
            similar_disorders.append(match.group())
            # print(match.group())
    return similar_disorders

def find_similar_disorders_caseSensitive(disorder,unique_labels):
    similar_disorders = []
    for label in unique_labels:
        match = re.search(".*"+disorder+".*", label)
        if match is not None:
            similar_disorders.append(match.group())
            # print(match.group())
    return similar_disorders

def update_row(row,similar_disorders,disorder):
    # print("Before:", row)
    new_label_list = []
    for item in row:
        if item in similar_disorders:
            new_label_list.append(disorder)
        else:
            new_label_list.append(item)
    # print("After:",new_label_list)
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

def concat_cols(impression, findings):
    if impression is np.nan:
        return findings
    elif findings is np.nan:
        return impression
    else:
        return findings+impression

def openI_dataset_preprocess(dataset_args):
    #------- download and set up openI dataset
    DATA_DIR = dataset_args['root']

    if os.path.exists(f"{DATA_DIR}/OpenI"):
        #---------load dataframe
        print("[  dataset  ] OpenI directory already exists")
        #--------load dataframe
        train_df_orig = pd.read_csv(f"{DATA_DIR}/{dataset_args['data_path']}/openI_train.csv")
        train_df_orig.loc[:,'labels'] = train_df_orig.apply(lambda row: np.array(ast.literal_eval(row['labels'])), axis=1)
        
        test_df = pd.read_csv(f"{DATA_DIR}/{dataset_args['data_path']}/openI_test.csv")
        test_df.loc[:,'labels'] = test_df.apply(lambda row: np.array(ast.literal_eval(row['labels'])), axis=1)

        mlflowLogger.store_param("dataset.len", len(train_df_orig)+len(test_df))

        #--------loading and storing labels to mlflow
        labels = list(np.load(f"{DATA_DIR}/OpenI/labels.npy"))
        num_labels = len(labels)
        mlflowLogger.store_param("col_names", labels)
        mlflowLogger.store_param("num_labels", num_labels)
        return train_df_orig, test_df, labels, num_labels

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
        # openI_df.to_csv(f'{DATA_DIR}/OpenI/openI.csv', index=False)
    
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
                'Nodule', 'surgical instrument']

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
        print("openI",labels)
        np.save(f"{DATA_DIR}/OpenI/labels.npy", list(labels))
        num_labels = len(labels)
        mlflowLogger.store_param("col_names", labels)
        mlflowLogger.store_param("num_labels", num_labels)

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

        #-------remove XXXX in text
        df_cls['text'] = df_cls.apply(lambda row: row.text.replace("XXXX", ""), axis=1)

        #-------shuffle
        df_cls = df_cls.sample(frac=1).reset_index(drop=True)
        mlflowLogger.store_param("dataset.len", len(df_cls))

        #-------stratified sampling
        train_indexes, test_indexes = iterative_train_test_split(df_cls['text'], np.array(df_cls['labels'].to_list()), 0.2)
        train_df = df_cls.iloc[train_indexes,:]
        test_df = df_cls.iloc[test_indexes,:]

        #-------save the datafarme
        df_cls_tosave = df_cls.copy()
        df_cls_tosave['labels'] = df_cls_tosave.apply(lambda row: list(row["labels"]), axis=1)
        df_cls_tosave.to_csv(f'{DATA_DIR}/OpenI/openI.csv', index=False)

        train_df_tosave = train_df.copy()
        train_df_tosave['labels'] = train_df_tosave.apply(lambda row: list(row["labels"]), axis=1)
        train_df_tosave.to_csv(f'{DATA_DIR}/OpenI/openI_train.csv', index=False)
        
        test_df_tosave = test_df.copy()
        test_df_tosave['labels'] = test_df_tosave.apply(lambda row: list(row["labels"]), axis=1)
        test_df_tosave.to_csv(f'{DATA_DIR}/OpenI/openI_test.csv', index=False)
        
        return train_df, test_df, labels



def _setup_dataset(dataset_name, dataset_args, tokenizer, tokenizer_args, head_type, head_args, batch_size, model_cfg): 
    
    train_df_orig, test_df, labels, num_labels = openI_dataset_preprocess(dataset_args)

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
    train_df_orig, test_df, labels, num_labels = openI_dataset_preprocess(dataset_args)
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

def _setup_dataset_ttest(dataset_name, dataset_args, tokenizer, tokenizer_args, head_type, head_args, batch_size, model_cfg, fold):
    """
    A function to create dataset splits for statistical test of MTL and GMTL
    """
    train_df_orig, test_df, labels, num_labels = openI_dataset_preprocess(dataset_args)

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

        train_dataloader_gmtl, validation_dataloader_gmtl, test_dataloader_gmtl, num_labels_gmtl = preprocess_cv(train_df, test_df, val_df, tokenizer, tokenizer_args, labels, labels_dict, "GMTL", head_args, num_labels, model_cfg, batch_size, fold_i)
        train_dataloader_mtl, validation_dataloader_mtl, test_dataloader_mtl, num_labels_mtl = preprocess_cv(train_df, test_df, val_df, tokenizer, tokenizer_args, labels, labels_dict, "MTL", head_args, num_labels, model_cfg, batch_size, fold_i)
        yield train_dataloader_gmtl, validation_dataloader_gmtl, test_dataloader_gmtl, num_labels_gmtl, train_dataloader_mtl, validation_dataloader_mtl, test_dataloader_mtl, num_labels_mtl

labels_dict={
    'Airspace Disease': "Airspace disease can be acute or chronic and commonly present as consolidation or ground-glass opacity on chest imaging. Consolidation or ground-glass opacity occurs when alveolar air is replaced by fluid, pus, blood, cells, or other material.", 
    'Atelectasis': "Atelectasis is a condition in which the airways and air sacs in the lung collapse or do not expand properly. Atelectasis can happen when there is an airway blockage, when pressure outside the lung keeps it from expanding, or when there is not enough surfactant for the lung to expand normally.", 
    'Calcified granuloma': "A calcified granuloma is a specific type of tissue inflammation that has become calcified over time. When something is referred to as “calcified,” it means that it contains deposits of the element calcium. Calcium has a tendency to collect in tissue that is healing.", 
    'Calcinosis': "Calcinosis is a condition in which abnormal amounts of calcium salts are found in soft body tissue, such as muscle.", 
    'Cardiomegaly': "An enlarged heart (cardiomegaly) isn't a disease, but rather a sign of another condition. The term cardiomegaly refers to an enlarged heart seen on any imaging test, including a chest X-ray.", 
    'Cicatrix': "Cicatrix is new tissue that forms over a wound and later contracts into a scar. ", 
    'Deformity': "Alteration in or distortion of the natural form of a part, organ, or the entire body. It may be acquired or congenital. If present after injury, deformity usually implies the presence of bone fracture, bone dislocation, or both.", 
    'Effusion': "Effusion. Chest X-rays can detect pleural effusions, which often appear as white areas at the lung base. A pleural effusion is a buildup of fluid in the pleural space, an area between the layers of tissue that line the lungs and the chest wall. It may also be referred to as effusion or pulmonary effusion.", 
    'Emphysema': "Emphysema is a lung condition that causes shortness of breath. In people with emphysema, the air sacs in the lungs (alveoli) are damaged.=", 
    'Medical Device': "Medical Device in chest", 
    'Nodule': "A nodule is a growth of abnormal tissue. Nodules can develop just below the skin. They can also develop in deeper skin tissues or internal organs.", 
    'indwelling catheters': "indwelling catheters is permanently present flexible tube inserted through a narrow opening into a body cavity, particularly the bladder, for removing fluid.", 
    'granulomatous disease': "A granuloma is a small area of inflammation. Granulomas are often found incidentally on an X-ray or other imaging test done for a different reason. Typically, granulomas are noncancerous (benign). Granulomas frequently occur in the lungs, but can occur in other parts of the body and head as well.", 
    'lung hyperdistention': "Hyperinflated lungs occur when air gets trapped in the lungs and causes them to overinflate. Hyperinflated lungs can be caused by blockages in the air passages or by air sacs that are less elastic, which interferes with the expulsion of air from the lungs.", 
    'lung hypoinflation': "lung hypoinflation can be caused by blockages in the air passages or by air sacs that are less elastic, which interferes with the expulsion of air from the lungs. ", 
    'opacity': "opacity. Lung opacities are vague, fuzzy clouds of white in the darkness of the lungs, which makes detecting them a real challenge.", 
    'spine degenerative': "spine degenerative. Degenerative changes in the spine are those that cause the loss of normal structure and/or function. They are not typically due to a specific injury but rather to age.", 
    'surgical instrument': "A surgical instrument is a tool or device for performing specific actions or carrying out desired effects during a surgery or operation, such as modifying biological tissue, or to provide access for viewing it.", 
    'thoracic vertebrae degenerative': "The thoracic spine is the area of your spine below your neck connected to your ribs. It is made up of 12 vertebral bodies with intervening discs. The disc is a cartilage, gristle like material that sits between your vertebral bodies in all parts of your spine. The disc acts like a cushion and also allows flexibility in your spine. The disc can wear over time as we age or earlier if it is injured. As it wears, or degenerates, the space between the vertebral bodies is reduced."
}