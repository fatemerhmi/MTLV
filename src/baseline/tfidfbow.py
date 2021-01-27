from skmultilearn.model_selection import IterativeStratification
import pandas as pd 
import numpy as np
import nltk
from tqdm import tqdm
import ast

from utils import preprocess, hamming_score, calculate_scores

from sklearn.metrics import accuracy_score, f1_score, hamming_loss

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier


def baselines(df_train_orig, df_test, algorithm = "randomForest", embd="bagofwords"):

    df_train_orig.loc[:, 'labels'] = df_train_orig.apply(lambda row: ast.literal_eval(row['labels']), axis=1)
    df_test.loc[:,'labels'] = df_test.apply(lambda row: ast.literal_eval(row['labels']), axis=1)

    df_train_orig.text = df_train_orig.text.apply(preprocess)
    df_test.text = df_test.text.apply(preprocess)

    c_vect = CountVectorizer(max_df=0.8, min_df=10, max_features=40000, ngram_range=(1,1),
                        lowercase=False)
    fold_i =0
    stratifier = IterativeStratification(n_splits=5, order=2)
    results = []
    for train_indexes, val_indexes in tqdm(stratifier.split(df_train_orig['text'], np.array(df_train_orig['labels'].to_list()))):
        fold_i += 1
        # print(f"[dataset] Fold {fold_i}")

        train_df_cv = df_train_orig.iloc[train_indexes,:]
        val_df_cv = df_train_orig.iloc[val_indexes,:]

        len_train_df_cv = len(train_df_cv)
        len_val_df_cv = len(val_df_cv)
        if embd == "bagofwords":
            vect = CountVectorizer(max_df=0.8, min_df=10, max_features=40000, ngram_range=(1,1),
                                lowercase=False)
        elif embd == "tfidf":
            vect = TfidfVectorizer(max_df=0.8, min_df=10, max_features=40000, ngram_range=(1,1),
                            lowercase=False)
                            
        X = c_vect.fit_transform(train_df_cv.text.to_list()+val_df_cv.text.to_list()+df_test.text.to_list()).toarray()
        X_train = X[:len_train_df_cv,:]
        X_val   = X[len_train_df_cv:len_train_df_cv+len_val_df_cv,:]
        X_test  = X[len_train_df_cv+len_val_df_cv:,:]

        y_train = train_df_cv.labels.to_list()
        y_val   = val_df_cv.labels.to_list()
        y_test  = df_test.labels.to_list()

        # model = RandomForestClassifier(max_depth=12, random_state=0)
        if algorithm == "randomForest":
            model = RandomForestClassifier(n_estimators=30, max_depth=500,
                                  min_samples_split=2,
                                  min_samples_leaf=1, max_leaf_nodes=None,
                                  class_weight='balanced')
            classifier = MultiOutputClassifier(model).fit(X_train, y_train)
        elif algorithm == "logisticRegression":
            model = LogisticRegression(random_state=0)
            classifier = MultiOutputClassifier(model).fit(X_train, y_train)
        elif algorithm == "xgboost":
            model = XGBClassifier(n_jobs=-1, max_depth=4)
            classifier = OneVsRestClassifier(model).fit(X_train, y_train)


        y_pred = classifier.predict(X_test)
        f1_score_micro, f1_score_macro, hamming_loss_, hamming_score_, subset_accuracy = calculate_scores(y_pred, y_test)

        results.append([f1_score_micro, f1_score_macro, hamming_loss_, hamming_score_, subset_accuracy])

    results = np.array(results)
    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)

    print()
    print("f1_micro =                  ", round(mean[0],2)," +- ", round(std[0],2))
    print("f1_macro =                  ", round(mean[1],2)," +- ", round(std[1],2))
    print("Hamming Loss =              ", round(mean[2],2)," +- ", round(std[2],2))
    print("Hamming Score (Accuracy) =  ", round(mean[3],2)," +- ", round(std[3],2))
    print("Subset Accuracy (sklearn) = ", round(mean[4],2)," +- ", round(std[4],2))