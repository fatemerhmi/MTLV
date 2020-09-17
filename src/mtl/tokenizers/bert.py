from transformers import BertTokenizer

__all__ = ['bert_base_uncased', 'bert_large_uncased', 'bert_base_cased', 'bert_large_cased']

def bert_base_uncased():
    return BertTokenizer.from_pretrained('bert-base-uncased')
    
def bert_large_uncased():
    return BertTokenizer.from_pretrained('bert-base-uncased')
    
def bert_base_cased():
    return BertTokenizer.from_pretrained('bert-base-cased')
    
def bert_large_cased():
    return BertTokenizer.from_pretrained('bert-large-cased')


#TODO: add fast tokenizer as well