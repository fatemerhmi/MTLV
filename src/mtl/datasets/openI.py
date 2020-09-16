import logging
import torch
import io
import os
from tqdm import tqdm

from mtl.datasets.utils import download_from_url, extract_archive, unicode_csv_reader

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
            text = ' '.join(row[1:])
            yield int(row[0]) - 1, text

def _create_data(iterator):
    data = []
    labels = []
    with tqdm(unit_scale=0, unit='lines') as t:
        for cls, text in iterator:
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

    @property
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


def _setup_datasets(dataset_name, tokenizer, tokenizer_args, root='.data'):
    # check not to download if it already exists
    if os.path.exists(f'{root}/{dataset_name}'): 
        #check if already preprocessed:
        if os.path.exists(f'{root}/{dataset_name}/cheXpertLabels'):

            train_csv_path = f"{root}/{dataset_name}/cheXpertLabels/test.csv"
            print('[ _setup_dataset ] Creating training data')
            train_data, train_labels = _create_data(_csv_iterator(train_csv_path))

            test_csv_path = f"{root}/{dataset_name}/cheXpertLabels/test.csv"
            print('[ _setup_dataset ] Creating testing data')
            test_data, test_labels = _create_data(_csv_iterator(test_csv_path))
            
            val_csv_path = f"{root}/{dataset_name}/cheXpertLabels/test.csv"
            print('[ _setup_dataset ] Creating testing data')
            val_data, val_labels = _create_data(_csv_iterator(val_csv_path))


    else: #dataset does not exists
        # download the datast:
        dataset_tar = download_from_url(URLS[dataset_name], root=root)

        # preprocess the dataset:

    extracted_files = extract_archive(dataset_tar)

    for fname in extracted_files:
        if fname.endswith('train.csv'):
            train_csv_path = fname
        if fname.endswith('test.csv'):
            test_csv_path = fname

    logging.info('Creating training data')
    train_data, train_labels = _create_data(_csv_iterator(train_csv_path))

    logging.info('Creating testing data')
    test_data, test_labels = _create_data(_csv_iterator(test_csv_path))

    if len(set(train_labels) ^ set(test_labels)) > 0:
        raise ValueError("Training and test labels don't match")

    return (TextClassification(train_data, train_labels, tokenizer, tokenizer_args),
            TextClassification(test_data, test_labels, tokenizer, tokenizer_args))


# def OpenI(*args, **kwargs): #lower
#     """ Defines AG_NEWS datasets.
#         The labels includes:
#             - 0 : World
#             - 1 : Sports
#             - 2 : Business
#             - 3 : Sci/Tech
#     Create supervised learning dataset: AG_NEWS
#     Separately returns the training and test dataset
#     Arguments:
#         root: Directory where the datasets are saved. Default: ".data"
#         ngrams: a contiguous sequence of n items from s string text.
#             Default: 1
#         vocab: Vocabulary used for dataset. If None, it will generate a new
#             vocabulary based on the train data set.
#         include_unk: include unknown token in the data (Default: False)
#     Examples:
#         >>> train_dataset, test_dataset = torchtext.datasets.AG_NEWS(ngrams=3)
#     """

#     return _setup_datasets(*(("AG_NEWS",) + args), **kwargs)