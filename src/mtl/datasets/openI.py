import logging
import torch
import io
import os
from tqdm import tqdm
import ast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pandas as pd
import numpy as np


from mtl.datasets.utils import download_from_url, extract_archive, unicode_csv_reader
from mtl.heads.utils import padding_heads, group_heads
import mtl.utils.logger as mlflowLogger 

"""
    You can manually donwload the OpenIdataset and put it in the data directory in root. 
    Description of the dataset: https://openi.nlm.nih.gov/faq
    Downlaod from: https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz

    If it's not not already in the data/OpenI directory, we will download it for you
"""

def _csv_iterator(data_path): # TODO: move this one to _create_data fater everything worked
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        for row in reader:
            if row[0]=='text' and row[1]=="labels":
                continue
            # print(row)
            text = row[0]
            label = ast.literal_eval(row[1]) 
            yield text, label

def _create_data(iterator):
    data = []
    labels = []
    with tqdm(unit_scale=0, unit='lines') as t:
        for text, cls in iterator:
            data.append((cls, text))
            labels.append(cls)
            t.update(1)
    return data, labels

class TextClassification(torch.utils.data.Dataset):

    def __init__(self,data, labels, tokenizer,tokenizer_args): 
        """Initiate OpenI dataset.
        Arguments:
            vocab: Vocabulary object used for dataset.
            data: a list of label/tokens tuple. tokens are a tensor after
                numericalizing the string tokens. label is an integer.
                [(label1, tokens1), (label2, tokens2), (label2, tokens3)]
            label: a set of the labels.
                {label1, label2}
        """

        super(TextClassification, self).__init__()
        self._data = data
        self._labels = labels
        self.tokenizer = tokenizer
        self.tokenizer_args = tokenizer_args

    def __getitem__(self, i):
        sentecne = self._data[i][1]
        inputs = self.tokenizer(sentecne, return_tensors="pt", **self.tokenizer_args)
        return (inputs, self._labels[i])

    # @property
    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for x in self._data:
            sentecne = x[0]
            inputs = self.tokenizer(sentecne, return_tensors="pt", **self.tokenizer_args)
            yield (inputs, x[1])

    @property
    def get_labels(self):
        return set(self._labels)
    
    @property
    def targets(self):
        return self._labels

def _setup_datasets_old(dataset_name, tokenizer, tokenizer_args, root='./data'):
    # check not to download if it already exists
    # import ipdb
    # ipdb.set_trace()
    if os.path.exists(f'{root}/{dataset_name}'): 
        #check if already preprocessed:
        if os.path.exists(f'{root}/{dataset_name}/cheXpertLabels'):

            train_csv_path = f"{root}/{dataset_name}/cheXpertLabels/train.csv"
            print('[ _setup_dataset ] Creating training data')
            train_data, train_labels = _create_data(_csv_iterator(train_csv_path))

            test_csv_path = f"{root}/{dataset_name}/cheXpertLabels/test.csv"
            print('[ _setup_dataset ] Creating testing data')
            test_data, test_labels = _create_data(_csv_iterator(test_csv_path))
            
            val_csv_path = f"{root}/{dataset_name}/cheXpertLabels/val.csv"
            print('[ _setup_dataset ] Creating testing data')
            val_data, val_labels = _create_data(_csv_iterator(val_csv_path))

            if len(set(train_labels[0]) ^ set(test_labels[0])) > 0:
                raise ValueError("Training and test labels don't match")

            return (TextClassification(train_data, train_labels, tokenizer, tokenizer_args),
                    TextClassification(val_data, val_labels, tokenizer, tokenizer_args),
                    TextClassification(test_data, test_labels, tokenizer, tokenizer_args))

    else: #dataset does not exists
        # download the datast:
        dataset_tar = download_from_url("https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz", root=root)

        # preprocess the dataset:
        print("[ openI dataset ] TODO: Finish this part")
        # extracted_files = extract_archive(dataset_tar)

        # for fname in extracted_files:
        #     if fname.endswith('train.csv'):
        #         train_csv_path = fname
        #     if fname.endswith('test.csv'):
        #         test_csv_path = fname

        # logging.info('Creating training data')
        # train_data, train_labels = _create_data(_csv_iterator(train_csv_path))

        # logging.info('Creating testing data')
        # test_data, test_labels = _create_data(_csv_iterator(test_csv_path))

        # if len(set(train_labels) ^ set(test_labels)) > 0:
        #     raise ValueError("Training and test labels don't match")

        # return (TextClassification(train_data, train_labels, tokenizer, tokenizer_args),
        #         TextClassification(test_data, test_labels, tokenizer, tokenizer_args))

#-------------this is the previous version I had in my jupyter notebook
def create_dataLoader(input, labels, batch_size):
    data = TensorDataset(input.input_ids, input.attention_mask, labels)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader


def _setup_datasets(dataset_name, dataset_args, tokenizer, tokenizer_args, head_type, head_args, batch_size): # previously read_data
    #-------read the data
    data_path =  f"{dataset_args['root']}/{dataset_args['data_path']}"        
    split_path = f"{dataset_args['root']}/{dataset_args['split_path']}"  

    df = pd.read_csv(data_path)
    cols = df.columns
    label_cols = list(cols[6:])
    num_labels = len(label_cols)

    #-------log how many labels + label cols
    mlflowLogger.store_param("col_names", label_cols)
    mlflowLogger.store_param("num_labels", num_labels)

    #-------convert string of list to actual list (labels)
    train_df = pd.read_csv(f"{split_path}/train.csv")
    train_df['labels'] = train_df.apply(lambda row: ast.literal_eval(row['labels']), axis=1)

    test_df = pd.read_csv(f"{split_path}/test.csv")
    test_df['labels'] = test_df.apply(lambda row: ast.literal_eval(row['labels']), axis=1)

    val_df = pd.read_csv(f"{split_path}/val.csv")
    val_df['labels'] = val_df.apply(lambda row: ast.literal_eval(row['labels']), axis=1)
    
    #-------check for multi-head
    if head_type =="multi-head":
        heads_index = head_args["heads_index"]
        mlflowLogger.store_param("heads_index", heads_index)
        padded_heads = padding_heads(heads_index)
        train_df = group_heads(padded_heads, train_df)
        test_df = group_heads(padded_heads, test_df)
        val_df = group_heads(padded_heads, val_df)
    if head_type =="single":
        print("TODO: [openI.py] Develope single head")
        return
    if head_type=="multi-task": 
        print("TODO: [openI.py] Develope multi-task")
        return

    #-------tokenize
    reports_train = train_df.text.to_list()
    reports_test = test_df.text.to_list()
    reports_val   = val_df.text.to_list()
   
    train = tokenizer(reports_train, padding=tokenizer_args['padding'], truncation=tokenizer_args['truncation'], max_length=tokenizer_args['max_length'], return_tensors="pt")
    test = tokenizer(reports_test, padding=tokenizer_args['padding'], truncation=tokenizer_args['truncation'], max_length=tokenizer_args['max_length'], return_tensors="pt")
    val = tokenizer(reports_val, padding=tokenizer_args['padding'], truncation=tokenizer_args['truncation'], max_length=tokenizer_args['max_length'], return_tensors="pt")
    
    #-------prepare labels for dataloader
    train_labels = torch.from_numpy(np.array(train_df.head_labels.to_list()))
    test_labels = torch.from_numpy(np.array(test_df.head_labels.to_list()))
    val_labels = torch.from_numpy(np.array(val_df.head_labels.to_list()))
    
    #-------create dataloarders
    train_dataloader      = create_dataLoader(train, train_labels, batch_size)
    validation_dataloader = create_dataLoader(val, val_labels, batch_size)
    test_dataloader       = create_dataLoader(test, test_labels, batch_size)

    return train_dataloader, validation_dataloader, test_dataloader