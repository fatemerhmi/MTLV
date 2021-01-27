import click
import pandas as pd
from tfidfbow import baselines
import numpy as np

def prepare_news_dataset(df_train, df_test):
    df_train.replace(np.nan, "", inplace=True)
    df_test.replace(np.nan, "", inplace=True)

    # df_train['labels'] = df_train.apply(lambda row: ast.literal_eval(row['labels']), axis=1)
    # df_test['labels'] = df_test.apply(lambda row: ast.literal_eval(row['labels']), axis=1)

    df_train['text'] = df_train.apply(lambda row: row['title']+" "+ row['desc'], axis=1)
    df_test['text'] = df_test.apply(lambda row: row['title']+" "+ row['desc'], axis=1)

    # remove columns except title and labels
    columns_to_remove = list(df_train.columns)
    columns_to_remove.remove("text")
    columns_to_remove.remove("labels")
    # print(columns_to_remove)
    df_train.drop(columns= columns_to_remove, inplace=True)
    df_test.drop(columns= columns_to_remove, inplace=True)
    return df_train, df_test

@click.group()
def main():
    """The package help is as follows."""
    pass

@main.command("run")
@click.option('--dataset', '-d', type=str, help='The name of the dataset, choose between: "openI", "news", "twentynewsgroup", "ohsumed", "reuters"')
@click.option('--classifier', '-c', type=str, help='Name of the classifier, choose between: "randomForest", "logisticRegression", "xgboost"')
@click.option('--embedding', '-e', type=str, help='Embedding approach, choose between: "tfidf", "bow"')
def run(dataset, classifier, embedding):
    if dataset == "openI":
        df_train = pd.read_csv("./data/OpenI/openI_train.csv") 
        df_test = pd.read_csv("./data/OpenI/openI_test.csv")
    elif dataset == "news":
        df_train = pd.read_csv("./data/news/train.csv") 
        df_test = pd.read_csv("./data/news/test.csv")
        df_train, df_test= prepare_news_dataset(df_train, df_test)
    else:
        df_train = pd.read_csv(f"./data/{dataset}/{dataset}_train.csv") 
        df_test = pd.read_csv(f"./data/{dataset}/{dataset}_test.csv")

    baselines(df_train, df_test, classifier, embedding)    
# print("------------OpenI:-----------------")
# df_train = pd.read_csv("./data/OpenI/openI_train.csv") 
# df_test = pd.read_csv("./data/OpenI/openI_test.csv")

# print("TFIDF + Random Forest")
# tfidf(df_train, df_test, "randomForest")

# print("TFIDF + Logistic Regression")
# tfidf(df_train, df_test, "logisticRegression")

# print("TFIDF + xgboost")
# tfidf(df_train, df_test, "xgboost")

# print("------------ohsumed:------------")
# df_train = pd.read_csv("./data/ohsumed/ohsumed_train.csv") 
# df_test = pd.read_csv("./data/ohsumed/ohsumed_test.csv")

# print("TFIDF + Random Forest")
# tfidf(df_train, df_test, "randomForest")

# print("------------reuters:------------")
# df_train = pd.read_csv("./data/reuters/reuters_train.csv") 
# df_test = pd.read_csv("./data/reuters/reuters_test.csv")
# tfidf(df_train, df_test)

# print("------------twentynewsgroup:------------")
# df_train = pd.read_csv("./data/twentynewsgroup/twentynewsgroup_train.csv") 
# df_test = pd.read_csv("./data/twentynewsgroup/twentynewsgroup_test.csv")
# tfidf(df_train, df_test)

if __name__ == "__main__":
    main()