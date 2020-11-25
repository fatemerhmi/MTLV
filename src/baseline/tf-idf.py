from skmultilearn.model_selection import IterativeStratification
import pandas as pd 
import numpy as np
import nltk
# from autocorrect import Speller
import ast

from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier


from utils import preprocess


def tfidf(df_train_orig, df_test):
    df_train_orig['labels'] = df_train_orig.apply(lambda row: ast.literal_eval(row['labels']), axis=1)
    df_test['labels'] = df_test.apply(lambda row: ast.literal_eval(row['labels']), axis=1)

    df_train_orig.text = df_train_orig.text.apply(preprocess)
    df_test.text = df_test.text.apply(preprocess)

    # Building a TF IDF matrix 
    # td = TfidfVectorizer(max_features = 8000) 
    td = TfidfVectorizer(max_df=0.8, min_df=10, max_features=40000, ngram_range=(1,1),
                        lowercase=False)
    fold_i =0
    stratifier = IterativeStratification(n_splits=5, order=2)
    results = []
    for train_indexes, val_indexes in stratifier.split(df_train_orig['text'], np.array(df_train_orig['labels'].to_list())):
        fold_i += 1
        print(f"[dataset] Fold {fold_i}")

        train_df_cv = df_train_orig.iloc[train_indexes,:]
        val_df_cv = df_train_orig.iloc[val_indexes,:]

        len_train_df_cv = len(train_df_cv)
        len_val_df_cv = len(val_df_cv)

        X = td.fit_transform(train_df_cv.text.to_list()+val_df_cv.text.to_list()+df_test.text.to_list()).toarray()
        X_train = X[:len_train_df_cv,:]
        X_val   = X[len_train_df_cv:len_train_df_cv+len_val_df_cv,:]
        X_test  = X[len_train_df_cv+len_val_df_cv:,:]

        y_train = train_df_cv.labels.to_list()
        y_val   = val_df_cv.labels.to_list()
        y_test  = df_test.labels.to_list()

        # from sklearn.naive_bayes import MultinomialNB
        # model = MultinomialNB()

        # model = RandomForestClassifier(max_depth=12, random_state=0)
        model = RandomForestClassifier(n_estimators=30, max_depth=500,
                               min_samples_split=2,
                               min_samples_leaf=1, max_leaf_nodes=None,
                               class_weight='balanced')
        classifier = MultiOutputClassifier(model).fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_micro = f1_score(y_test, y_pred,average='micro')
        f1_macro = f1_score(y_test, y_pred,average='macro')

        results.append([f1_micro,f1_macro,accuracy])

    results = np.array(results)
    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)

    print("f1_micro= ", mean[0]*100," +- ", std[0]*100)
    print("f1_macro = ", mean[1]*100," +- ", std[1]*100)
    print("acc = ", mean[2]*100," +- ", std[2]*100)

print("------------OpenI:-----------------")
df_train = pd.read_csv("./data/OpenI/openI_train.csv") 
df_test = pd.read_csv("./data/OpenI/openI_test.csv")
tfidf(df_train, df_test)

print("------------ohsumed:------------")
df_train = pd.read_csv("./data/ohsumed/ohsumed_train.csv") 
df_test = pd.read_csv("./data/ohsumed/ohsumed_test.csv")
tfidf(df_train, df_test)

print("------------reuters:------------")
df_train = pd.read_csv("./data/reuters/reuters_train.csv") 
df_test = pd.read_csv("./data/reuters/reuters_test.csv")
tfidf(df_train, df_test)

print("------------twentynewsgroup:------------")
df_train = pd.read_csv("./data/twentynewsgroup/twentynewsgroup_train.csv") 
df_test = pd.read_csv("./data/twentynewsgroup/twentynewsgroup_test.csv")
tfidf(df_train, df_test)

