import nltk
import re
import string
nltk.download('punkt')
from nltk.tokenize import word_tokenize as wt

nltk.download('stopwords')
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