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
import numpy as np
from skmultilearn.model_selection import IterativeStratification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from prettytable import PrettyTable

from mtl.heads.grouping_KDE import *
from mtl.heads.grouping_meanshift import *
from mtl.heads.grouping_kmediod import grouping_kmediod, get_all_label_embds, plot_elbow_method, plot_emb_groups
import mtl.utils.configuration as configuration
from mtl.heads.utils import padding_heads, group_heads
import mtl.utils.logger as mlflowLogger 

#------from torch text-----------
def download_from_url(url, path=None, root='.data', overwrite=False, hash_value=None,
                      hash_type="sha256"):
    """Download file, with logic (from tensor2tensor) for Google Drive. Returns
    the path to the downloaded file.
    Arguments:
        url: the url of the file from URL header. (None)
        root: download folder used to store the file in (.data)
        overwrite: overwrite existing files (False)
        hash_value (str, optional): hash for url (Default: ``None``).
        hash_type (str, optional): hash type, among "sha256" and "md5" (Default: ``"sha256"``).
    Examples:
        >>> url = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
        >>> torchtext.utils.download_from_url(url)
        >>> url = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
        >>> torchtext.utils.download_from_url(url)
        >>> '.data/validation.tar.gz'
    """
    def _check_hash(path):
        if hash_value:
            with open(path, "rb") as file_obj:
                if not validate_file(file_obj, hash_value, hash_type):
                    raise RuntimeError("The hash of {} does not match. Delete the file manually and retry.".format(path))

    def _process_response(r, root, filename):
        chunk_size = 16 * 1024
        total_size = int(r.headers.get('Content-length', 0))
        if filename is None:
            d = r.headers['content-disposition']
            filename = re.findall("filename=\"(.+)\"", d)
            if filename is None:
                raise RuntimeError("Filename could not be autodetected")
            filename = filename[0]
        path = os.path.join(root, filename)
        if os.path.exists(path):
            logging.info('File %s already exists.' % path)
            if not overwrite:
                _check_hash(path)
                return path
            logging.info('Overwriting file %s.' % path)
        logging.info('Downloading file {} to {}.'.format(filename, path))
        with open(path, "wb") as file:
            with tqdm(total=total_size, unit='B',
                      unit_scale=1, desc=path.split('/')[-1]) as t:
                for chunk in r.iter_content(chunk_size):
                    if chunk:
                        file.write(chunk)
                        t.update(len(chunk))
        logging.info('File {} downloaded.'.format(path))

        _check_hash(path)
        return path

    if path is None:
        _, filename = os.path.split(url)
    else:
        root, filename = os.path.split(path)

    if not os.path.exists(root):
        try:
            os.makedirs(root)
        except OSError:
            print("Can't create the download directory {}.".format(root))
            raise

    if 'drive.google.com' not in url:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True)
        return _process_response(response, root, filename)
    else:
        # google drive links get filename from google drive
        filename = None

    logging.info('Downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    return _process_response(response, root, filename)

def extract_archive(from_path, to_path=None, overwrite=False):
    """Extract archive.
    Arguments:
        from_path: the path of the archive.
        to_path: the root path of the extracted files (directory of from_path)
        overwrite: overwrite existing files (False)
    Returns:
        List of paths to extracted files even if not overwritten.
    Examples:
        >>> url = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
        >>> from_path = './validation.tar.gz'
        >>> to_path = './'
        >>> torchtext.utils.download_from_url(url, from_path)
        >>> torchtext.utils.extract_archive(from_path, to_path)
        >>> ['.data/val.de', '.data/val.en']
        >>> torchtext.utils.download_from_url(url, from_path)
        >>> torchtext.utils.extract_archive(from_path, to_path)
        >>> ['.data/val.de', '.data/val.en']
    """

    if to_path is None:
        to_path = os.path.dirname(from_path)

    if from_path.endswith(('.tar.gz', '.tgz')):
        logging.info('Opening tar file {}.'.format(from_path))
        with tarfile.open(from_path, 'r') as tar:
            files = []
            for file_ in tar:
                file_path = os.path.join(to_path, file_.name)
                if file_.isfile():
                    files.append(file_path)
                    if os.path.exists(file_path):
                        logging.info('{} already extracted.'.format(file_path))
                        if not overwrite:
                            continue
                tar.extract(file_, to_path)
            return files

    elif from_path.endswith('.zip'):
        assert zipfile.is_zipfile(from_path), from_path
        logging.info('Opening zip file {}.'.format(from_path))
        with zipfile.ZipFile(from_path, 'r') as zfile:
            files = []
            for file_ in zfile.namelist():
                file_path = os.path.join(to_path, file_)
                files.append(file_path)
                if os.path.exists(file_path):
                    logging.info('{} already extracted.'.format(file_path))
                    if not overwrite:
                        continue
                zfile.extract(file_, to_path)
        files = [f for f in files if os.path.isfile(f)]
        return files

    elif from_path.endswith('.gz'):
        default_block_size = 65536
        filename = from_path[:-3]
        files = [filename]
        with gzip.open(from_path, 'rb') as gzfile, \
                open(filename, 'wb') as d_file:
            while True:
                block = gzfile.read(default_block_size)
                if not block:
                    break
                else:
                    d_file.write(block)
            d_file.write(block)
        return files

    else:
        raise NotImplementedError(
            "We currently only support tar.gz, .tgz, .gz and zip achives.")

def unicode_csv_reader(unicode_csv_data, **kwargs):
    r"""Since the standard csv library does not handle unicode in Python 2, we need a wrapper.
    Borrowed and slightly modified from the Python docs:
    https://docs.python.org/2/library/csv.html#csv-examples
    Arguments:
        unicode_csv_data: unicode csv data (see example below)
    Examples:
        >>> from torchtext.utils import unicode_csv_reader
        >>> import io
        >>> with io.open(data_path, encoding="utf8") as f:
        >>>     reader = unicode_csv_reader(f)
    """

    # Fix field larger than field limit error
    maxInt = sys.maxsize
    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)
    csv.field_size_limit(maxInt)

    for line in csv.reader(unicode_csv_data, **kwargs):
        yield line

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
    if head_type =="multi-task":
        #check the type:   
        if head_args['type'] == "givenset":
            heads_index = head_args["heads_index"]

        elif head_args['type'] == "KDE":
            print("[  dataset  ] KDE label grouping starts!")
            heads_index = KDE(label_counts_train, head_args['bandwidth'])

        elif head_args['type'] == "meanshift":
            print("[  dataset  ] meanshift label grouping starts!")
            heads_index = meanshift(label_counts_train)

        elif head_args['type'] == "kmediod-label":
            print("[  dataset  ] kmediod-label grouping starts!")
            # model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
            model = configuration.setup_model(model_cfg)(num_labels, "emb_cls")
            embds = get_all_label_embds(labels, tokenizer, model)
            if "elbow" in head_args.keys():
                plot_elbow_method(embds,head_args['elbow'])
            heads_index, cluster_label = grouping_kmediod(embds, head_args['clusters'])
            if "plot" in head_args.keys():
                if head_args['plot'] == True:
                    plot_emb_groups(embds, labels, cluster_label)
            del model

        elif head_args['type'] == "kmediod-labeldesc":
            print("[  dataset  ] kmediod-label description grouping starts!")
            # model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
            model = configuration.setup_model(model_cfg)(num_labels, "emb_cls")
            labels_list = [labels_dict[label] for label in labels]
            # list(labels_dict.values()
            embds = get_all_label_embds(labels_list, tokenizer, model)
            if "elbow" in head_args.keys():
                plot_elbow_method(embds,head_args['elbow'])
            heads_index, cluster_label = grouping_kmediod(embds, head_args['clusters'])
            if "plot" in head_args.keys():
                if head_args['plot'] == True:
                    plot_emb_groups(embds, labels, cluster_label)
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

    # inputs = self.tokenizer.encode_plus(
    #         text,
    #         None,
    #         add_special_tokens=True,
    #         max_length=self.max_len,
    #         padding='max_length',
    #         return_token_type_ids=True,
    #         truncation=True,
    #     )


    # train = tokenizer.encode_plus(reports_train, \
    #     None, \
    #     add_special_tokens=True, \
    #     max_length=tokenizer_args['max_length'], \
    #     padding=tokenizer_args['padding'], \
    #     return_token_type_ids=True, \
    #     truncation=tokenizer_args['truncation'], \
    #     return_tensors="pt")

    # test = tokenizer.encode_plus(reports_test, \
    #     None, \
    #     add_special_tokens=True, \
    #     max_length=tokenizer_args['max_length'], \
    #     padding=tokenizer_args['padding'], \
    #     return_token_type_ids=True, \
    #     truncation=tokenizer_args['truncation'], \
    #     return_tensors="pt")

    # val = tokenizer.encode_plus(reports_val, \
    #     None, \
    #     add_special_tokens=True, \
    #     max_length=tokenizer_args['max_length'], \
    #     padding=tokenizer_args['padding'], \
    #     return_token_type_ids=True, \
    #     truncation=tokenizer_args['truncation'], \
    #     return_tensors="pt")

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

    #-------check for multi-head, single or multi-task
    if head_type =="multi-task":
        #check the type:   
        if head_args['type'] == "givenset":
            heads_index = head_args["heads_index"]

        elif head_args['type'] == "KDE":
            print("[  dataset  ] KDE label grouping starts!")
            heads_index = KDE(label_counts_train, head_args['bandwidth'])

        elif head_args['type'] == "meanshift":
            print("[  dataset  ] meanshift label grouping starts!")
            heads_index = meanshift(label_counts_train)

        elif head_args['type'] == "kmediod-label":
            print("[  dataset  ] kmediod-label grouping starts!")
            # model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
            model = configuration.setup_model(model_cfg)(num_labels, "emb_cls")
            embds = get_all_label_embds(labels, tokenizer, model)
            if "elbow" in head_args.keys():
                plot_elbow_method(embds,head_args['elbow'])
            heads_index, cluster_label = grouping_kmediod(embds, head_args['clusters'])
            if "plot" in head_args.keys():
                if head_args['plot'] == True:
                    plot_emb_groups(embds, labels, cluster_label)
            del model

        elif head_args['type'] == "kmediod-labeldesc":
            print("[  dataset  ] kmediod-label description grouping starts!")
            # model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
            model = configuration.setup_model(model_cfg)(num_labels, "emb_cls")
            labels_list = [labels_dict[label] for label in labels]
            # list(labels_dict.values()
            embds = get_all_label_embds(labels_list, tokenizer, model)
            if "elbow" in head_args.keys():
                plot_elbow_method(embds,head_args['elbow'])
            heads_index = grouping_kmediod(embds, head_args['clusters'])
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