from transformers import BertTokenizer
import os

__all__ = ['bert_base_uncased', 'bert_large_uncased', 'bert_base_cased', 'bert_large_cased']

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


#TODO: add fast tokenizer 