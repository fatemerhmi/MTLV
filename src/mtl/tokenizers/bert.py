from transformers import BertTokenizer
import os

__all__ = ['bert_base_uncased', 'bert_large_uncased', \
           'bert_base_cased', 'bert_large_cased',  \
            'BlueBERT_Base', "BioBERT_Basev1_1" ]
        #    'BioBERT_Basev1_1', 'BioBERT_Basev1_0_PM', \
        #     'BioBERT_Basev1_0_PMC', 'BioBERT_Basev1_0_PM_PMC']

def bert_base_uncased():
    MODEL_PATH = "model_weights/bert-base-uncased"
    if not os.path.exists(f"{MODEL_PATH}"):
        return BertTokenizer.from_pretrained("bert-base-uncased")
    else:
        return BertTokenizer.from_pretrained(MODEL_PATH)
    
def bert_base_cased():
    MODEL_PATH = "model_weights/bert-base-cased"
    if not os.path.exists(f"{MODEL_PATH}"):
        return BertTokenizer.from_pretrained("bert-base-cased")
    else:
        return BertTokenizer.from_pretrained(MODEL_PATH)

def bert_large_uncased():
    return BertTokenizer.from_pretrained('bert-large-uncased')
    
def bert_large_cased():
    return BertTokenizer.from_pretrained('bert-large-cased')


def BioBERT_Basev1_1():
    MODEL_PATH = "model_weights/biobert-v1.1"
    MODEL_NAME = 'dmis-lab/biobert-v1.1'

    if not os.path.exists(f"{MODEL_PATH}"):
        return BertTokenizer.from_pretrained(MODEL_NAME)
    else:
        return BertTokenizer.from_pretrained(MODEL_PATH)

def BlueBERT_Base():
    MODEL_PATH = "model_weights/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
    MODEL_NAME = "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"

    if not os.path.exists(MODEL_PATH):
        return BertTokenizer.from_pretrained(MODEL_NAME)
    else:
        return BertTokenizer.from_pretrained(MODEL_PATH)

# def BioBERT_Basev1_0_PM():
#     MODEL_PATH = "model_weights/biobert_v1.0_pubmed"
#     return BertTokenizer.from_pretrained(MODEL_PATH, return_dict=True)

# def BioBERT_Basev1_0_PMC():
#     MODEL_PATH = "model_weights/biobert_v1.0_pmc"
#     return BertTokenizer.from_pretrained(MODEL_PATH, return_dict=True)

# def BioBERT_Basev1_0_PM_PMC():
#     MODEL_PATH = "model_weights/biobert_v1.0_pubmed_pmc"
#     return BertTokenizer.from_pretrained(MODEL_PATH, return_dict=True)
