from skmultilearn.model_selection import IterativeStratification
from prettytable import PrettyTable
import os
from pyunpack import Archive
import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
import torch
from transformers import BertModel

from mtl.datasets.utils import iterative_train_test_split, create_dataLoader, create_new_column
from mtl.heads.utils import padding_heads, group_heads
import mtl.utils.logger as mlflowLogger 
from mtl.datasets.utils import iterative_train_test_split, create_dataLoader
from mtl.heads.grouping_KDE import *
from mtl.heads.grouping_meanshift import *
from mtl.heads.grouping_kmediod import grouping_kmediod, get_all_label_embds, plot_elbow_method

def _setup_datasets(dataset_name, dataset_args, tokenizer, tokenizer_args, head_type, head_args, batch_size):
    #------- load or download then set up ohsumed dataset
    DATA_DIR = dataset_args['root']

    if os.path.exists(f"{DATA_DIR}/{dataset_args['data_path']}"):
        print("[  dataset  ] ohsumed directory already exists.")

        #--------load dataframe
        train_df = pd.read_csv(f"{DATA_DIR}/{dataset_args['data_path']}/ohsumed_train.csv")
        train_df['labels'] = train_df.apply(lambda row: np.array(ast.literal_eval(row['labels'])), axis=1)
        
        test_df = pd.read_csv(f"{DATA_DIR}/{dataset_args['data_path']}/ohsumed_test.csv")
        test_df['labels'] = test_df.apply(lambda row: np.array(ast.literal_eval(row['labels'])), axis=1)
        mlflowLogger.store_param("dataset.len", len(train_df)+len(test_df))
        #--------loading and storing labels to mlflow
        labels = list(np.load(f"{DATA_DIR}/ohsumed/labels.npy"))
        num_labels = len(labels)
        mlflowLogger.store_param("col_names", labels)
        mlflowLogger.store_param("num_labels", num_labels)

    else:
        os.makedirs(f"{DATA_DIR}/ohsumed")
        
        print("[  dataset  ] ohsumed dataset is being downloaded...")
        os.system(f'wget -N -P {DATA_DIR}/ohsumed http://disi.unitn.eu/moschitti/corpora/ohsumed-all-docs.tar.gz')
        os.system(f'wget -N -P {DATA_DIR}/ohsumed http://disi.unitn.eu/moschitti/corpora/First-Level-Categories-of-Cardiovascular-Disease.txt')
        
        print("[  dataset  ] Extracting openI dataset...")
        directory_to_extract_to = f"{DATA_DIR}/ohsumed/"
        Archive(f"{DATA_DIR}/ohsumed/ohsumed-all-docs.tar.gz").extractall(directory_to_extract_to)
        os.system(f"rm {DATA_DIR}/ohsumed/ohsumed-all-docs.tar.gz")

        #------storing label details to mlflow and npy file
        labels = os.listdir (f"{DATA_DIR}/ohsumed/ohsumed-all")
        if ".DS_Store" in labels:
            labels.remove(".DS_Store")
            
        np.save(f"{DATA_DIR}/ohsumed/labels.npy", labels)
        num_labels = len(labels)
        mlflowLogger.store_param("col_names", labels)
        mlflowLogger.store_param("num_labels", num_labels)

        #------convert the files to a dataframe 
        all_data = []
        for label in tqdm(labels):
            instances_in_a_label = os.listdir (f"{DATA_DIR}/ohsumed/ohsumed-all/{label}")
            for item in instances_in_a_label:
                f = open(f"{DATA_DIR}/ohsumed/ohsumed-all/{label}/{item}", "r")
                raw_data = f.read()
                all_data.append([item, raw_data, label])
        all_data = np.asarray(all_data)
        df = pd.DataFrame(all_data, columns=["id", "text", "label"])

        os.system(f"rm -r {DATA_DIR}/ohsumed/ohsumed-all")

        #------preprocessing the labels
        tqdm.pandas()
        print("\n[  dataset  ] ohsumed preprocessing of labels begin...")
        df["labels"] = df.progress_apply(lambda row: df.loc[df['id'] == row['id']].label.tolist(), axis=1)

        #------remove duplicate rows
        df.drop_duplicates('id', inplace=True)

        #------bring labels to seperate columns
        for label in tqdm(labels):
            df = create_new_column(df, label)

        df.drop(['labels', "label"],inplace=True, axis=1)

        df['labels'] = df.apply(lambda row: np.array(row[labels].to_list()), axis=1)
        df.drop(labels, inplace=True, axis=1)
        #-------shuffle
        df = df.sample(frac=1).reset_index(drop=True)
        mlflowLogger.store_param("dataset.len", len(df))

        #-------stratified sampling
        train_indexes, test_indexes = iterative_train_test_split(df['text'], np.array(df['labels'].to_list()), 0.2)

        train_df = df.iloc[train_indexes,:]
        test_df = df.iloc[test_indexes,:]

        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        
        #-------save the datafarme
        train_df_tosave = train_df.copy()
        train_df_tosave['labels'] = train_df_tosave.apply(lambda row: list(row["labels"]), axis=1)
        train_df_tosave.to_csv(f'{DATA_DIR}/ohsumed/ohsumed_train.csv', index=False)
        del train_df_tosave
        
        test_df_tosave = test_df.copy()
        test_df_tosave['labels'] = test_df_tosave.apply(lambda row: list(row["labels"]), axis=1)
        test_df_tosave.to_csv(f'{DATA_DIR}/ohsumed/ohsumed_test.csv', index=False)
        del test_df_tosave

    train_indexes, val_indexes = iterative_train_test_split(train_df['text'], np.array(train_df['labels'].to_list()), 0.15)

    val_df = train_df.iloc[val_indexes,:]
    train_df = train_df.iloc[train_indexes,:]

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    print('Train: ', len(train_df))
    mlflowLogger.store_param("dataset.train.len", len(train_df))
    print('Test: ', len(test_df))
    mlflowLogger.store_param("dataset.test.len", len(test_df))
    print('Val: ', len(val_df))
    mlflowLogger.store_param("dataset.val.len", len(val_df))

    #-------table for train, test,val counts
    # label_counts_total = np.array(df.labels.to_list()).sum(axis=0)
    label_counts_train = np.array(train_df.labels.to_list()).sum(axis=0)
    label_counts_test = np.array(test_df.labels.to_list()).sum(axis=0)
    label_counts_val = np.array(val_df.labels.to_list()).sum(axis=0)

    pretty=PrettyTable()
    label_counts_total = []
    pretty.field_names = ['Label', 'total', 'train', 'test','val']
    for pathology, cnt_train, cnt_test, cnt_val in zip(labels, label_counts_train, label_counts_test, label_counts_val):
        cnt_total = cnt_train + cnt_test + cnt_val
        label_counts_total.append(cnt_total)
        pretty.add_row([pathology, cnt_total, cnt_train, cnt_test, cnt_val])
    print(pretty)

    # pretty=PrettyTable()
    # pretty.field_names = ['Label', 'total', 'train', 'test','val']
    # for pathology, cnt_total, cnt_train, cnt_test, cnt_val in zip(labels, label_counts_total, label_counts_train, label_counts_test, label_counts_val):
    #     pretty.add_row([pathology, cnt_total, cnt_train, cnt_test, cnt_val])
    # print(pretty)

    #-------check for multi-head, single or multi-task
    if head_type =="multi-task":
        #check the type:   label_counts_total
        if head_args['type'] == "givenset":
            heads_index = head_args["heads_index"]
        elif head_args['type'] == "KDE":
            heads_index = KDE(label_counts_total, head_args['bandwidth'])
        elif head_args['type'] == "meanshift":
             heads_index = meanshift(label_counts_total)
        elif head_args['type'] == "kmediod-label":
            print("[  dataset  ] kmediod-label grouping starts!")
            model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
            embds = get_all_label_embds(labels, tokenizer, model) # change this one to use setup model function
            if "elbow" in head_args.keys():
                plot_elbow_method(embds,head_args['elbow'])
            heads_index = grouping_kmediod(embds, head_args['clusters'])
            del model
        elif head_args['type'] == "kmediod-labeldesc":
            model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
            labels_list = [labels_dict[labels_C_to_title[label]] for label in labels]
            embds = get_all_label_embds(labels_list, tokenizer, model)
            if "elbow" in head_args.keys():
                plot_elbow_method(embds,head_args['elbow'])
            heads_index = grouping_kmediod(embds, head_args['clusters'])
            del model

        mlflowLogger.store_param("heads_index", heads_index)
        padded_heads = padding_heads(heads_index)
        
        #--group the heads
        train_df = group_heads(padded_heads, train_df)
        test_df = group_heads(padded_heads, test_df)
        val_df = group_heads(padded_heads, val_df)

        #--prepare labels for dataloader
        train_labels = torch.from_numpy(np.array(train_df.head_labels.to_list()))
        test_labels = torch.from_numpy(np.array(test_df.head_labels.to_list()))
        val_labels = torch.from_numpy(np.array(val_df.head_labels.to_list()))

    elif head_type =="single-head":
        #--prepare labels for dataloader
        train_labels = torch.from_numpy(np.array(train_df.labels.to_list()))
        test_labels = torch.from_numpy(np.array(test_df.labels.to_list()))
        val_labels = torch.from_numpy(np.array(val_df.labels.to_list()))
    else:
        raise Exception("The head type must be either 'multi-task' or 'single-head'!")

    #-------tokenize
    reports_train = train_df.text.to_list()
    reports_test = test_df.text.to_list()
    reports_val   = val_df.text.to_list()
   
    train = tokenizer(reports_train, padding=tokenizer_args['padding'], truncation=tokenizer_args['truncation'], max_length=tokenizer_args['max_length'], return_tensors="pt")
    test = tokenizer(reports_test, padding=tokenizer_args['padding'], truncation=tokenizer_args['truncation'], max_length=tokenizer_args['max_length'], return_tensors="pt")
    val = tokenizer(reports_val, padding=tokenizer_args['padding'], truncation=tokenizer_args['truncation'], max_length=tokenizer_args['max_length'], return_tensors="pt")
    
    #-------create dataloarders
    train_dataloader      = create_dataLoader(train, train_labels, batch_size)
    validation_dataloader = create_dataLoader(val, val_labels, batch_size)
    test_dataloader       = create_dataLoader(test, test_labels, batch_size)

    return train_dataloader, validation_dataloader, test_dataloader, num_labels

#============labels================
labels_C_to_title = {
    'C01': 'Bacterial Infections and Mycoses',
    'C02': 'Virus Diseases', 
    'C03': 'Parasitic Diseases', 
    'C04': 'Neoplasms', 
    'C05': 'Musculoskeletal Diseases', 
    'C06': 'Digestive System Diseases', 
    'C07': 'Stomatognathic Diseases', 
    'C08': 'Respiratory Tract Diseases', 
    'C09': 'Otorhinolaryngologic Diseases', 
    'C10': 'Nervous System Diseases', 
    'C11': 'Eye Diseases', 
    'C12': 'Urologic and Male Genital Diseases', 
    'C13': 'Female Genital Diseases and Pregnancy Complications', 
    'C14': 'Cardiovascular Diseases', 
    'C15': 'Hemic and Lymphatic Diseases', 
    'C16': 'Neonatal Diseases and Abnormalities', 
    'C17': 'Skin and Connective Tissue Diseases', 
    'C18': 'Nutritional and Metabolic Diseases', 
    'C19': 'Endocrine Diseases', 
    'C20': 'Immunologic Diseases', 
    'C21': 'Disorders of Environmental Origin', 
    'C22': 'Animal Diseases', 
    'C23': 'Pathological Conditions, Signs and Symptoms'
    }

labels_dict={
    'Bacterial Infections and Mycoses': "Bacterial Infections and Mycoses. A bacterial infection is a proliferation of a harmful strain of bacteria on or inside the body. Mycoses are common and a variety of environmental and physiological conditions can contribute to the development of fungal diseases. Inhalation of fungal spores or localized colonization of the skin may initiate persistent infections; therefore, mycoses often start in the lungs or on the skin.",
    'Virus Diseases': "Virus Diseases. Viruses cause familiar infectious diseases such as the common cold, flu and warts. ", 
    'Parasitic Diseases': "A parasitic disease, also known as parasitosis, is an infectious disease caused or transmitted by a parasite. Many parasites do not cause diseases as it may eventually lead to death of both organism and host. Parasites infecting human beings are called human parasites.", 
    'Neoplasms': "A neoplasm is an abnormal growth of cells, also known as a tumor. Neoplastic diseases are conditions that cause tumor growth — both benign and malignant. Benign tumors are noncancerous growths.",
    'Musculoskeletal Diseases': "Musculoskeletal disorders (MSD) are injuries or disorders that affect the human body’s movement or musculoskeletal system. such as muscles, nerves, tendons, joints, cartilage, and spinal discs. ",
    'Digestive System Diseases': "Digestive System Diseases. A digestive disease is any health problem that occurs in the digestive tract. Conditions may range from mild to serious. Some common problems include heartburn, cancer, irritable bowel syndrome, and lactose intolerance. Other digestive diseases include: Gallstones, cholecystitis, and cholangitis.", 
    'Stomatognathic Diseases': "General or unspecified diseases of the stomatognathic system, comprising the mouth, teeth, jaws, and pharynx. Synonym(s): Dental Diseases, Mouth and Tooth Diseases, Dental Disease, Disease, Dental, Narrow term(s): Jaw Diseases.", 
    'Respiratory Tract Diseases': "Respiratory Tract Diseases. A type of disease that affects the lungs and other parts of the respiratory system. Respiratory diseases may be caused by infection, by smoking tobacco, or by breathing in secondhand tobacco smoke, radon, asbestos, or other forms of air pollution.", 
    'Otorhinolaryngologic Diseases': "Otorhinolaryngologic Diseases is a branch of medicine that deals with diagnosis and treatment of diseases of the ear, nose, and throat.", 
    'Nervous System Diseases': "Nervous system diseases, any of the diseases or disorders that affect the functioning of the human nervous system. Everything that humans sense, consider, and effect and all the unlearned reflexes of the body depend on the functioning of the nervous system.", 
    'Eye Diseases': "Eye Diseases are any disease of the eye or cornea. macular degeneration. eye disease caused by degeneration of the cells of the macula lutea and results in blurred vision; can cause blindness. retinopathy. a disease of the retina that can result in loss of vision.", 
    'Urologic and Male Genital Diseases': "Urologic and Male Genital Diseases. A male genital disease is a condition that affects the male reproductive system. An example is orchitis.",
    'Female Genital Diseases and Pregnancy Complications': "Female Genital Diseases and Pregnancy Complications, Vaginal Diseases, Complications of pregnancy are health problems that occur during pregnancy. They can involve the mother's health, the baby's health, or both. Some women have health problems that arise during pregnancy, and other women have health problems before they become pregnant that could lead to complications.",
    'Cardiovascular Diseases': "Cardiovascular disease (CVD) is the name for the group of disorders of heart and blood vessels, and include: hypertension (high blood pressure) coronary heart disease (heart attack) cerebrovascular disease (stroke)", 
    'Hemic and Lymphatic Diseases': "Hemic and Lymphatic Diseases. Hemic diseases include disorders involving the formed elements (e.g., Erythrocyte Aggregation, Intravascular) and chemical components (e.g., BLOOD PROTEIN DISORDERS); lymphatic diseases include disorders relating to lymph, lymph nodes, and lymphocytes.", 
    'Neonatal Diseases and Abnormalities': "Neonatal Diseases and Abnormalities. Diseases existing at birth and often before birth, or that develop during the first month of life (Infant, Newborn, Diseases), regardless of causation." , 
    'Skin and Connective Tissue Diseases': "A connective tissue disease is any disease that affects the parts of the body that connect the structures of the body together. Connective tissues are made up of two proteins: collagen and elastin.", 
    'Nutritional and Metabolic Diseases': "Nutritional and Metabolic Diseases. A metabolic disorder occurs when the metabolism process fails and causes the body to have either too much or too little of the essential substances needed to stay healthy. Our bodies are very sensitive to errors in metabolism. The body must have amino acids and many types of proteins to perform all of its functions.", 
    'Endocrine Diseases': "Endocrine Diseases. Endocrine disorders are diseases related to the endocrine glands of the body. The endocrine system produces hormones, which are chemical signals sent out, or secreted, through the bloodstream.", 
    'Immunologic Diseases': "Immunologic Diseases. Immunological disorders are diseases or conditions caused by a dysfunction of the immune system and include allergy, asthma, autoimmune diseases, autoinflammatory syndromes and immunological deficiency syndromes.", 
    'Disorders of Environmental Origin': "Disorders of Environmental Origin. Disorders caused by external forces rather than by physiologic dysfunction or by pathogens.  ", 
    'Animal Diseases': "Animal diseases, an impairment of the normal state of an animal that interrupts or modifies its vital functions.", 
    'Pathological Conditions, Signs and Symptoms' : "Pathological Conditions, Signs and Symptoms. Abnormal anatomical or physiological conditions and objective or subjective manifestations of disease, not classified as disease or syndrome."
}