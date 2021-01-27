import nltk
import re
import string
import numpy as np
from sklearn import metrics

# nltk.download('punkt')
from nltk.tokenize import word_tokenize as wt

# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def preprocess(row):
    # lower case
    row = row.lower()

    #remove symbols
    row = row.translate(str.maketrans('', '', string.punctuation))

    #remove non alphabetic 
    row = re.sub('[^A-Za-z]', ' ', row)

    #tokenize
    row = wt(row)
    
    #remove words with less than 3 characters
    row = [word for word in row if len(word)>2]

    # remove stop words and stemming
    # stemmer = PorterStemmer()
    # row = [stemmer.stem(word) for word in row if word not in set(stopwords.words('english'))]
    row = [word for word in row if word not in set(stopwords.words('english'))]

    #if a word appears too much remove it

    #join
    row = " ".join(row)

    return row


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)



def calculate_scores(outputs, targets):
    outputs = np.array(outputs) >= 0.5
    
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_micro = round(f1_score_micro*100,2)

    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    f1_score_macro = round(f1_score_macro*100,2)

    hamming_loss_ = metrics.hamming_loss(targets, outputs)
    hamming_loss_ = round(hamming_loss_*100,2)

    hamming_score_ = hamming_score(np.array(targets), np.array(outputs))
    hamming_score_ = round(hamming_score_*100,2)

    subset_accuracy = metrics.accuracy_score(targets, outputs)
    subset_accuracy = round(subset_accuracy*100,2)

    return f1_score_micro, f1_score_macro, hamming_loss_, hamming_score_, subset_accuracy