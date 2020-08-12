import logging
import torch
import io
import os
from src.mtl.datasets.utils import download_from_url, extract_archive, unicode_csv_reader
from tqdm import tqdm

URLS = {
    'AG_NEWS':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms',
    'SogouNews':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUkVqNEszd0pHaFE',
    'DBpedia':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k',
    'YelpReviewPolarity':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbNUpYQ2N3SGlFaDg',
    'YelpReviewFull':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0',
    'YahooAnswers':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU',
    'AmazonReviewPolarity':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM',
    'AmazonReviewFull':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZVhsUnRWRDhETzA'
}

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
    """Defines an abstract text classification datasets.
       Currently, we only support the following datasets:
             - AG_NEWS
             - SogouNews
             - DBpedia
             - YelpReviewPolarity
             - YelpReviewFull
             - YahooAnswers
             - AmazonReviewPolarity
             - AmazonReviewFull
    """

    def __init__(self,data, labels, tokenizer,tokenizer_args): 
        """Initiate text-classification dataset.
        Arguments:
            vocab: Vocabulary object used for dataset.
            data: a list of label/tokens tuple. tokens are a tensor after
                numericalizing the string tokens. label is an integer.
                [(label1, tokens1), (label2, tokens2), (label2, tokens3)]
            label: a set of the labels.
                {label1, label2}
        Examples:
            See the examples in examples/text_classification/
        """

        super(TextClassification, self).__init__()
        self._data = data
        self._labels = labels
        self.tokenizer = tokenizer
        self.tokenizer_args = tokenizer_args

    # @property
    def __getitem__(self, i):
        sentecne = self._data[i][1]
        inputs = self.tokenizer(sentecne, return_tensors="pt", **self.tokenizer_args)
        return (inputs, self._labels[i])

    @property
    def __len__(self):
        return len(self._data)

    # @property
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
    if os.path.exists(f'{root}/{FILENAMES[dataset_name]}'): 
        dataset_tar = f'{root}/{FILENAMES[dataset_name]}'
    else: #dataset does not exists
        dataset_tar = download_from_url(URLS[dataset_name], root=root)
    
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

@register.setdatasetname("AG_NEWS") 
def AG_NEWS(*args, **kwargs): #lower
    """ Defines AG_NEWS datasets.
        The labels includes:
            - 0 : World
            - 1 : Sports
            - 2 : Business
            - 3 : Sci/Tech
    Create supervised learning dataset: AG_NEWS
    Separately returns the training and test dataset
    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)
    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.AG_NEWS(ngrams=3)
    """

    return _setup_datasets(*(("AG_NEWS",) + args), **kwargs)

@register.setdatasetname("SogouNews")
def SogouNews(*args, **kwargs):
    """ Defines SogouNews datasets.
        The labels includes:
            - 0 : Sports
            - 1 : Finance
            - 2 : Entertainment
            - 3 : Automobile
            - 4 : Technology
    Create supervised learning dataset: SogouNews
    Separately returns the training and test dataset
    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)
    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.SogouNews(ngrams=3)
    """

    return _setup_datasets(*(("SogouNews",) + args), **kwargs)

@register.setdatasetname("DBpedia")
def DBpedia(*args, **kwargs):
    """ Defines DBpedia datasets.
        The labels includes:
            - 0 : Company
            - 1 : EducationalInstitution
            - 2 : Artist
            - 3 : Athlete
            - 4 : OfficeHolder
            - 5 : MeanOfTransportation
            - 6 : Building
            - 7 : NaturalPlace
            - 8 : Village
            - 9 : Animal
            - 10 : Plant
            - 11 : Album
            - 12 : Film
            - 13 : WrittenWork
    Create supervised learning dataset: DBpedia
    Separately returns the training and test dataset
    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)
    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.DBpedia(ngrams=3)
    """

    return _setup_datasets(*(("DBpedia",) + args), **kwargs)

@register.setdatasetname("YelpReviewPolarity")
def YelpReviewPolarity(*args, **kwargs):
    """ Defines YelpReviewPolarity datasets.
        The labels includes:
            - 0 : Negative polarity.
            - 1 : Positive polarity.
    Create supervised learning dataset: YelpReviewPolarity
    Separately returns the training and test dataset
    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)
    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.YelpReviewPolarity(ngrams=3)
    """

    return _setup_datasets(*(("YelpReviewPolarity",) + args), **kwargs)

@register.setdatasetname("YelpReviewFull")
def YelpReviewFull(*args, **kwargs):
    """ Defines YelpReviewFull datasets.
        The labels includes:
            0 - 4 : rating classes (4 is highly recommended).
    Create supervised learning dataset: YelpReviewFull
    Separately returns the training and test dataset
    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)
    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.YelpReviewFull(ngrams=3)
    """

    return _setup_datasets(*(("YelpReviewFull",) + args), **kwargs)

@register.setdatasetname("YahooAnswers")
def YahooAnswers(*args, **kwargs):
    """ Defines YahooAnswers datasets.
        The labels includes:
            - 0 : Society & Culture
            - 1 : Science & Mathematics
            - 2 : Health
            - 3 : Education & Reference
            - 4 : Computers & Internet
            - 5 : Sports
            - 6 : Business & Finance
            - 7 : Entertainment & Music
            - 8 : Family & Relationships
            - 9 : Politics & Government
    Create supervised learning dataset: YahooAnswers
    Separately returns the training and test dataset
    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)
    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.YahooAnswers(ngrams=3)
    """

    return _setup_datasets(*(("YahooAnswers",) + args), **kwargs)

@register.setdatasetname("AmazonReviewPolarity")
def AmazonReviewPolarity(*args, **kwargs):
    """ Defines AmazonReviewPolarity datasets.
        The labels includes:
            - 0 : Negative polarity
            - 1 : Positive polarity
    Create supervised learning dataset: AmazonReviewPolarity
    Separately returns the training and test dataset
    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)
    Examples:
       >>> train_dataset, test_dataset = torchtext.datasets.AmazonReviewPolarity(ngrams=3)
    """

    return _setup_datasets(*(("AmazonReviewPolarity",) + args), **kwargs)

@register.setdatasetname("AmazonReviewFull")
def AmazonReviewFull(*args, **kwargs):
    """ Defines AmazonReviewFull datasets.
        The labels includes:
            0 - 4 : rating classes (4 is highly recommended)
    Create supervised learning dataset: AmazonReviewFull
    Separately returns the training and test dataset
    Arguments:
        root: Directory where the dataset are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)
    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.AmazonReviewFull(ngrams=3)
    """

    return _setup_datasets(*(("AmazonReviewFull",) + args), **kwargs)


DATASETS = {
    'AG_NEWS': AG_NEWS,
    'SogouNews': SogouNews,
    'DBpedia': DBpedia,
    'YelpReviewPolarity': YelpReviewPolarity,
    'YelpReviewFull': YelpReviewFull,
    'YahooAnswers': YahooAnswers,
    'AmazonReviewPolarity': AmazonReviewPolarity,
    'AmazonReviewFull': AmazonReviewFull
}


LABELS = {
    'AG_NEWS': {0: 'World',
                1: 'Sports',
                2: 'Business',
                3: 'Sci/Tech'},
    'SogouNews': {0: 'Sports',
                  1: 'Finance',
                  2: 'Entertainment',
                  3: 'Automobile',
                  4: 'Technology'},
    'DBpedia': {0: 'Company',
                1: 'EducationalInstitution',
                2: 'Artist',
                3: 'Athlete',
                4: 'OfficeHolder',
                5: 'MeanOfTransportation',
                6: 'Building',
                7: 'NaturalPlace',
                8: 'Village',
                9: 'Animal',
                10: 'Plant',
                11: 'Album',
                12: 'Film',
                13: 'WrittenWork'},
    'YelpReviewPolarity': {0: 'Negative polarity',
                           1: 'Positive polarity'},
    'YelpReviewFull': {0: 'score 1',
                       1: 'score 2',
                       2: 'score 3',
                       3: 'score 4',
                       4: 'score 5'},
    'YahooAnswers': {0: 'Society & Culture',
                     1: 'Science & Mathematics',
                     2: 'Health',
                     3: 'Education & Reference',
                     4: 'Computers & Internet',
                     5: 'Sports',
                     6: 'Business & Finance',
                     7: 'Entertainment & Music',
                     8: 'Family & Relationships',
                     9: 'Politics & Government'},
    'AmazonReviewPolarity': {0: 'Negative polarity',
                             1: 'Positive polarity'},
    'AmazonReviewFull': {0: 'score 1',
                         1: 'score 2',
                         2: 'score 3',
                         3: 'score 4',
                         4: 'score 5'}
}

FILENAMES = {
    'AG_NEWS' : "ag_news_csv.tar.gz"
}