import ast
import pandas as pd
from skmultilearn.model_selection import IterativeStratification
import numpy as np
import os 
import re

from mtl.datasets.utils import preprocess, preprocess_cv
from mtl.heads.utils import padding_heads, group_heads
import mtl.utils.logger as mlflowLogger 
from mtl.datasets.utils import iterative_train_test_split, create_dataLoader
from mtl.heads.grouping_KDE import *
from mtl.heads.grouping_meanshift import *
from mtl.heads.grouping_kmediod import grouping_kmediod, get_all_label_embds, plot_elbow_method
import mtl.utils.configuration as configuration

def prepare_text_col(df_train, df_test):
    df_train.replace(np.nan, "", inplace=True)
    df_test.replace(np.nan, "", inplace=True)

    df_train.loc[:,'labels'] = df_train.apply(lambda row: ast.literal_eval(row['labels']), axis=1)
    df_test.loc[:,'labels'] = df_test.apply(lambda row: ast.literal_eval(row['labels']), axis=1)

    df_train.loc[:,'text'] = df_train.apply(lambda row: row['title']+" "+ row['desc'], axis=1)
    df_test.loc[:,'text'] = df_test.apply(lambda row: row['title']+" "+ row['desc'], axis=1)

    # remove columns except title and labels
    columns_to_remove = list(df_train.columns)
    columns_to_remove.remove("text") 
    columns_to_remove.remove("labels")
    # print(columns_to_remove)
    df_train.drop(columns= columns_to_remove, inplace=True)
    df_test.drop(columns= columns_to_remove, inplace=True)

    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_test = df_test.sample(frac=1).reset_index(drop=True)

    return df_train, df_test

def news_dataset_preprocess(dataset_args):
    ##NOTE: put if statement here
    #------- download and set up openI dataset
    DATA_DIR = dataset_args['root']

    if os.path.exists(f"{DATA_DIR}/{dataset_args['data_path']}"):
        #---------load dataframe
        print("[  dataset  ] news directory already exists")
       
        train_df_orig = pd.read_csv(f"{DATA_DIR}/news/train.csv")
        test_df = pd.read_csv(f"{DATA_DIR}/news/test.csv")

        mlflowLogger.store_param("dataset.len", len(train_df_orig)+len(test_df))

        #--------loading and storing labels to mlflow
        cols = train_df_orig.columns
        labels = list(cols[3:-1])
        labels = [re.sub(r'_+', ' ', s) for s in labels]
        num_labels = len(labels)
        mlflowLogger.store_param("col_names", labels)
        mlflowLogger.store_param("num_labels", num_labels)

        #---------prepare the dataset
        train_df_orig, test_df = prepare_text_col(train_df_orig, test_df)

        return train_df_orig, test_df, labels, num_labels
    else:
        raise Exception(f"{DATA_DIR}/{dataset_args['data_path']} does not exists!")

def save_fold_train_validation(train_df, val_df, dataset_args, fold_i):
    DATA_DIR = dataset_args['root']

    train_df['labels'] = train_df.apply(lambda row: list(row["labels"]), axis=1)
    train_df.to_csv(f'{DATA_DIR}/news/train_fold{fold_i}.csv', index=False)
    
    val_df['labels'] = val_df.apply(lambda row: list(row["labels"]), axis=1)
    val_df.to_csv(f'{DATA_DIR}/news/validation_fold{fold_i}.csv', index=False)
    

def read_fold_train_validattion(fold_i, dataset_args):
    DATA_DIR = dataset_args['root']

    #---------load dataframe
    print("[  dataset  ] reading Fold:{fold_i} train and validation set.")
    
    #--------load dataframe
    train_df = pd.read_csv(f"{DATA_DIR}/news/train_fold{fold_i}.csv")
    val_df = pd.read_csv(f"{DATA_DIR}/news/validation_fold{fold_i}.csv")

    train_df.replace(np.nan, "", inplace=True)
    val_df.replace(np.nan, "", inplace=True)

    train_df.loc[:,'labels'] = train_df.labels.apply(ast.literal_eval)
    val_df.loc[:,'labels'] = val_df.labels.apply(ast.literal_eval)

    return train_df, val_df


def _setup_dataset(dataset_name, dataset_args, tokenizer, tokenizer_args, head_type, head_args, batch_size, model_cfg): 

    train_df_orig, test_df, labels, num_labels = news_dataset_preprocess(dataset_args)

    train_indexes, val_indexes = iterative_train_test_split(train_df_orig['text'], np.array(train_df_orig['labels'].to_list()), 0.15)
    val_df = train_df_orig.iloc[val_indexes,:]
    train_df = train_df_orig.iloc[train_indexes,:]

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    train_dataloader, validation_dataloader, test_dataloader, num_labels = preprocess(train_df, test_df, val_df, tokenizer, tokenizer_args, labels, labels_dict, head_type, head_args, num_labels, model_cfg, batch_size)
    return train_dataloader, validation_dataloader, test_dataloader, num_labels

def _setup_dataset_cv(dataset_name, dataset_args, tokenizer, tokenizer_args, head_type, head_args, batch_size, model_cfg, fold):
    DATA_DIR = dataset_args['root']
    train_df_orig, test_df, labels, num_labels = news_dataset_preprocess(dataset_args)
    fold_i =0
    test_df.reset_index(drop=True, inplace=True)

    stratifier = IterativeStratification(n_splits=fold, order=2)
    for train_indexes, val_indexes in stratifier.split(train_df_orig['text'], np.array(train_df_orig['labels'].to_list())):
        fold_i += 1
        print(f"[dataset] ======================================= Fold {fold_i} =======================================")
        if not os.path.exists(f'{DATA_DIR}/news/train_fold{fold_i}.csv'):
            print("&"*40)
            val_df = train_df_orig.iloc[val_indexes,:]
            train_df = train_df_orig.iloc[train_indexes,:]

            train_df.reset_index(drop=True, inplace=True)
            val_df.reset_index(drop=True, inplace=True)

            save_fold_train_validation(train_df, val_df, dataset_args, fold_i)
        else:
            print("$"*40)
            train_df, val_df = read_fold_train_validattion(fold_i, dataset_args)

        train_dataloader, validation_dataloader, test_dataloader, num_labels = preprocess_cv(train_df, test_df, val_df, tokenizer, tokenizer_args, labels, labels_dict, head_type, head_args, num_labels, model_cfg, batch_size, fold_i)
        yield train_dataloader, validation_dataloader, test_dataloader, num_labels

def _setup_dataset_ttest(dataset_name, dataset_args, tokenizer, tokenizer_args, head_type, head_args, batch_size, model_cfg, fold):

    train_df_orig, test_df, labels, num_labels = news_dataset_preprocess(dataset_args)

    fold_i =0
    stratifier = IterativeStratification(n_splits=fold, order=2)
    for train_indexes, val_indexes in stratifier.split(train_df_orig['text'], np.array(train_df_orig['labels'].to_list())):
        fold_i += 1
        print(f"[dataset] ======================================= Fold {fold_i} =======================================")

        val_df = train_df_orig.iloc[val_indexes,:]
        train_df = train_df_orig.iloc[train_indexes,:]

        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)

        
        train_dataloader_s, validation_dataloader_s, test_dataloader_s, num_labels_s = preprocess_cv(train_df, test_df, val_df, tokenizer, tokenizer_args, labels, labels_dict, "single-head", head_args, num_labels, model_cfg, batch_size, fold_i)
        train_dataloader_mtl, validation_dataloader_mtl, test_dataloader_mtl, num_labels_mtl = preprocess_cv(train_df, test_df, val_df, tokenizer, tokenizer_args, labels, labels_dict, "multi-task", head_args, num_labels, model_cfg, batch_size, fold_i)
        yield train_dataloader_s, validation_dataloader_s, test_dataloader_s, num_labels_s, train_dataloader_mtl, validation_dataloader_mtl, test_dataloader_mtl, num_labels_mtl

labels_dict={
    'new service launch': "A new service launch refers to a business's planned and coordinated effort to debut a new service to the market and make that product generally available for purchase. ",
    'executive change': "executive change or management change happens in organization or company",
    'award and recognition': "award an recognition in companies can be customer service award, above and beyond, peer-to-peer recognition, perfect attendence program or other type of awards",
    'new product launch': "new product launch is when a company launch or introduce a new product",
    'divestment': "divestment is the action or process of selling off subsidiary business interests or investments",
    'financial loss': "financial loss is loss of money or decrease in financial value",
    'customer loss growth decline': "customer loss is the loss of clients or customers. growth decline is when business does not keep pace with the rest of the country's economic growth, or when its rate of growth contracts across multiple measurement periods declines",
    'organizational restructuring': "organizational restructuring is the act of changing the business model of an organization to transform it for the better. These changes can be legal, operational processes, ownership, etc.",
    'merger acquisition': "mergers and acquisitions (M&A) refer broadly to the process of one company combining with one another.",
    'cost cutting': "cost cutting or Cost reduction is the process used by companies to reduce their costs and increase their profits.",
    'business shut-down': "A shutdown point is a level of operations at which a company experiences no benefit for continuing operations and therefore decides to shut down temporarily—or in some cases permanently.",
    'bankruptcy': "Bankruptcy is a legal proceeding involving a person or business that is unable to repay their outstanding debts",
    'regulatory settlement': "A regulatory settlement agreement is a formal written agreement between the Solicitors Regulation Authority (SRA) and a regulated person which settles complaints made by the SRA against the regulated person.",
    'regulatory investigation': "Regulatory investigation meansa formal hearing, official investigation, examination, inquiry, legal action or any other similar proceeding initiated by a governmental, regulatory, law enforcement, professional or statutory body against you or a company",
    'initial public offering': "An initial public offering (IPO) refers to the process of offering shares of a private corporation to the public in a new stock issuance. Public share issuance allows a company to raise capital from public investors.",
    'joint venture': "A joint venture is a business entity created by two or more parties, generally characterized by shared ownership, shared returns and risks, and shared governance.",
    'hiring': "hiring is the process of identifying, attracting, interviewing, suitable candidates for jobs within an organization or a company",
    'financial result': "The financial result is the difference between earnings before interest and taxes and earnings before taxes. It is determined by the earning or the loss which results from financial affairs.",
    'fundraising investment': "Investment is the action or process of investing money for profit or material result. Fundraising or fund-raising is the process of seeking and gathering voluntary financial contributions by engaging individuals, businesses, charitable foundations, or governmental agencies.",
    'executive appointment': "executive appointment is when a company appoint or hire a new executive",
    'regulatory approval': "Regulatory Approval means any approvals, product and or establishment licenses, registrations or authorizations of a company",
    'downsizing': "downsizing is making a company or organization smaller by eliminating staff positions or lay off current staff",
    'product shutdown': "product shutdown is when a company no longer provides service or updates or support a product",
    'pricing': "pricing is a change or decide the amount required as payment for a service or product",
    'new fund launch': "A new fund offer occurs when a new fund is launched, allowing the firm to raise capital for purchasing securities. Mutual funds are one of the most common new fund offerings marketed by an investment company.",
    'going private': "going private is when stockholder or other affiliated person that reduces the number of stockholders of a public company, allowing the company to terminate its public company status and become private.",
    'investment exit': " An exit occurs when an investor decides to get rid of their stake in a company. If an investor exits, then they will either have a profit or a loss. ",
    'employee dispute strike': "a complaint, argument, or disagreement between employees and their employer or between two or more employees. Strike is a work stoppage, caused by the mass refusal of employees to work.",
    'delist': "Delisting is the removal of a listed security from a stock exchange. The delisting of a security can be voluntary or involuntary and usually results when a company ceases operations, declares bankruptcy, merges, does not meet listing requirements, or seeks to become private.",
    'business outlook projections': "Business Outlook Surveys are qualitative surveys to track the current economic situation and to forecast short-term trends. Financial projections use existing or estimated financial data to forecast your business's future income and expenses.",
    'company executive statement': "company executive statement is the stement about status of the company said from one of the company executiv",
    'board decisions': "meeting with the board of directors to make a decision about the company",
    'business expansion': "Business Expansion is a stage where the business reaches the point for growth and seeks out for additional options to generate more profit.",
    'buyout': "a buyout is an investment transaction by which the ownership equity of a company, or a majority share of the stock of the company is acquired",
    'alliance partnership': "A partnership company is formed when the parties involved agree to share the business's profits or losses proportionately. A strategic alliance is an agreement between two or more parties to pursue a set of agreed upon objectives needed while remaining independent organizations. ",
    'law suit judgement settlement': "law suit judgement settlement, In law, a settlement is a resolution between disputing parties about a legal case, reached either before or after court action begins"
}