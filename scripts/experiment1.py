import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm, trange
from transformers import AdamW
from torch.nn import BCEWithLogitsLoss, BCELoss
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score, label_ranking_average_precision_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import ast


def read_data(data_path,split_path):
    df = pd.read_csv(data_path)

    cols = df.columns
    label_cols = list(cols[6:])
    

    train_df = pd.read_csv(f"{split_path}/train.csv")
    # convert str to list
    train_df['labels'] = train_df.apply(lambda row: ast.literal_eval(row['labels']), axis=1)

    test_df = pd.read_csv(f"{split_path}/test.csv")
    test_df['labels'] = test_df.apply(lambda row: ast.literal_eval(row['labels']), axis=1)

    val_df = pd.read_csv(f"{split_path}/val.csv")
    val_df['labels'] = val_df.apply(lambda row: ast.literal_eval(row['labels']), axis=1)

    return train_df, test_df, val_df, label_cols

def create_dataLoader(input, labels, batch_size):
    data = TensorDataset(input.input_ids, input.attention_mask, labels)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader


def multilabel_cls(data_path, split_path, PreTrainedModel, epochs, batch_size, max_length ,ModelTokenizer, tokenizer_name, model_name, use_data_loader):
    #prepare the dataset
    train_df, test_df, val_df,  label_cols = read_data(data_path, split_path)
    num_labels = len(label_cols)
    # ----------tokenize---------------
    tokenizer = ModelTokenizer.from_pretrained(tokenizer_name)

    reports_train = train_df.text.to_list()
    reports_test = test_df.text.to_list()
    reports_val   = val_df.text.to_list()

    train = tokenizer(reports_train, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
    test = tokenizer(reports_test, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
    val = tokenizer(reports_val, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")

    train_labels = torch.from_numpy(np.array(train_df.labels.to_list()))
    test_labels = torch.from_numpy(np.array(test_df.labels.to_list()))
    val_labels = torch.from_numpy(np.array(val_df.labels.to_list()))

    #-----------dataloaders--------------
    if use_data_loader: # if the dataset is huge in size
        # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, 
        # unlike a for loop, with an iterator the entire dataset does not need to be loaded into memory

        train_dataloader = create_dataLoader(train, train_labels, batch_size)
        validation_dataloader   = create_dataLoader(val, val_labels, batch_size)
        test_dataloader  = create_dataLoader(test, test_labels, batch_size)

    else: #TODO: if the dataset is small in size
        pass

    # Load model, the pretrained model will include a single linear classification layer on top for classification. 
    model = PreTrainedModel.from_pretrained(model_name, num_labels=num_labels)
    model.cuda()
    optimizer = AdamW(model.parameters(),lr=2e-5)  # Default optimization
    
    #---------FineTune model-----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device()
    # print(device)
    # device = torch.cuda.device(1)
    # print(device)
    # n_gpu = torch.cuda.device_count()
    print(torch.cuda.get_device_name(0))
    
    # Store our loss and accuracy for plotting
    train_loss_set = []

    # trange is a tqdm wrapper around the normal python range
    for _ in trange(epochs, desc="Epoch"):

        #-------Training-------

        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0 #running loss
        nb_tr_examples, nb_tr_steps = 0, 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()

            # # Forward pass for multiclass classification
            # outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            # loss = outputs[0]
            # logits = outputs[1]

            # Forward pass for multilabel classification
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]
            loss_func = BCEWithLogitsLoss() 
            loss = loss_func(logits.view(-1,num_labels),b_labels.type_as(logits).view(-1,num_labels)) #convert labels to float for calculation
            # loss_func = BCELoss() 
            # loss = loss_func(torch.sigmoid(logits.view(-1,num_labels)),b_labels.type_as(logits).view(-1,num_labels)) #convert labels to float for calculation
            train_loss_set.append(loss.item())    

            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # scheduler.step()
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss/nb_tr_steps))

        # ---------Validation--------

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Variables to gather full output
        logit_preds,true_labels,pred_labels,tokenized_texts = [],[],[],[]

        # Predict
        for i, batch in enumerate(validation_dataloader):
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                # Forward pass
                outs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                b_logit_pred = outs[0]
                pred_label = torch.sigmoid(b_logit_pred)

                b_logit_pred = b_logit_pred.detach().cpu().numpy()
                pred_label = pred_label.to('cpu').numpy()
                b_labels = b_labels.to('cpu').numpy()

            tokenized_texts.append(b_input_ids)
            logit_preds.append(b_logit_pred)
            true_labels.append(b_labels)
            pred_labels.append(pred_label)

        # Flatten outputs
        pred_labels = [item for sublist in pred_labels for item in sublist]
        true_labels = [item for sublist in true_labels for item in sublist]

        # Calculate Accuracy
        threshold = 0.50
        pred_bools = [pl>threshold for pl in pred_labels]
        true_bools = [tl==1 for tl in true_labels]
        val_f1_accuracy = f1_score(true_bools,pred_bools,average='micro')*100
        val_flat_accuracy = accuracy_score(true_bools, pred_bools)*100

        print('F1 Validation Accuracy: ', val_f1_accuracy)
        print('Validation Accuracy: ', val_flat_accuracy)

    # ---------test--------
    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    #track variables
    logit_preds,true_labels,pred_labels,tokenized_texts = [],[],[],[]

    # Predict
    for i, batch in enumerate(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            # Forward pass
            outs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            b_logit_pred = outs[0]
            pred_label = torch.sigmoid(b_logit_pred)

            b_logit_pred = b_logit_pred.detach().cpu().numpy()
            pred_label = pred_label.to('cpu').numpy()
            b_labels = b_labels.to('cpu').numpy()

        tokenized_texts.append(b_input_ids)
        logit_preds.append(b_logit_pred)
        true_labels.append(b_labels)
        pred_labels.append(pred_label)

    # Flatten outputs
    tokenized_texts = [item for sublist in tokenized_texts for item in sublist]
    pred_labels = [item for sublist in pred_labels for item in sublist]
    true_labels = [item for sublist in true_labels for item in sublist]
    # Converting flattened binary values to boolean values
    true_bools = [tl==1 for tl in true_labels]

    #We need to threshold our sigmoid function outputs which range from [0, 1]. Below I use 0.50 as a threshold.
    test_label_cols = label_cols
    pred_bools = [pl>0.50 for pl in pred_labels] #boolean output after thresholding

    # Print and save classification report
    print("-----------test-----------")
    print("Threshold: 0.5")
    print('Test F1 Score: ', f1_score(true_bools, pred_bools,average='micro'))
    print('Test Accuracy: ', accuracy_score(true_bools, pred_bools))
    print('LRAP: ', label_ranking_average_precision_score(true_labels, pred_labels) ,'\n')
    clf_report = classification_report(true_bools,pred_bools,target_names=test_label_cols)
    # pickle.dump(clf_report, open('classification_report.txt','wb')) #save report
    print(clf_report)

    # Calculate Accuracy - maximize F1 accuracy by tuning threshold values. First with 'macro_thresholds' on the order of e^-1 then with 'micro_thresholds' on the order of e^-2
    print("-----Optimizing threshold value for micro F1 score-----")

    macro_thresholds = np.array(range(1,10))/10

    f1_results, flat_acc_results = [], []
    for th in macro_thresholds:
        pred_bools = [pl>th for pl in pred_labels]
        test_f1_accuracy = f1_score(true_bools,pred_bools,average='micro')
        test_flat_accuracy = accuracy_score(true_bools, pred_bools)
        f1_results.append(test_f1_accuracy)
        flat_acc_results.append(test_flat_accuracy)

    best_macro_th = macro_thresholds[np.argmax(f1_results)] #best macro threshold value

    micro_thresholds = (np.array(range(10))/100)+best_macro_th #calculating micro threshold values

    f1_results, flat_acc_results = [], []
    for th in micro_thresholds:
        pred_bools = [pl>th for pl in pred_labels]
        test_f1_accuracy = f1_score(true_bools,pred_bools,average='micro')
        test_flat_accuracy = accuracy_score(true_bools, pred_bools)
        f1_results.append(test_f1_accuracy)
        flat_acc_results.append(test_flat_accuracy)

    best_f1_idx = np.argmax(f1_results) #best threshold value

    # Printing and saving classification report
    print('Best Threshold: ', micro_thresholds[best_f1_idx])
    print('Test F1 Score: ', f1_results[best_f1_idx])
    print('Test Accuracy: ', flat_acc_results[best_f1_idx])
    print('LRAP: ', label_ranking_average_precision_score(true_labels, pred_labels) , '\n')

    best_pred_bools = [pl>micro_thresholds[best_f1_idx] for pl in pred_labels]
    clf_report_optimized = classification_report(true_bools,best_pred_bools, target_names=label_cols)
    # pickle.dump(clf_report_optimized, open('classification_report_optimized.txt','wb'))
    print(clf_report_optimized)

def main():    
    data_path = 'data/OpenI/OpenI_cheXpertLabels.csv'
    split_path = 'data/OpenI/cheXpertLabels'
    use_data_loader = True
    
    epochs = 3 # Number of training epochs (authors recommend between 2 and 4)
    batch_size = 16
    max_length = 128
    #----------------bert---------------
    # from transformers import BertTokenizer, BertForSequenceClassification
    # model_name = 'bert-base-uncased'
    # tokenizer_name = "bert-base-uncased"
    
    # # model_name = "bert-base-cased"
    # # tokenizer_name = "bert-base-cased"
    # ModelTokenizer = BertTokenizer
    # PreTrainedModel = BertForSequenceClassification

    #----------------BioBERT-v1.0----------
    # model_name = "monologg/biobert_v1.0_pubmed_pmc" # from hugginface model list
    # model_name = "model_wieghts/biobert_v1.0_pubmed_pmc"
    # tokenizer_name = "bert-base-cased"
    # ModelTokenizer = BertTokenizer
    # PreTrainedModel = BertForSequenceClassification

    #----------------BioBERT-v1.1----------
    # model_name = "model_wieghts/biobert_v1.1_pubmed"
    # tokenizer_name = "bert-base-cased"
    # ModelTokenizer = BertTokenizer
    # PreTrainedModel = BertForSequenceClassification
			
    #------------roberta-------------
    # from transformers import RobertaTokenizer, RobertaForSequenceClassification
    # model_name = 'roberta-base'
    # tokenizer_name = 'roberta-base'
    # ModelTokenizer = RobertaTokenizer
    # PreTrainedModel = RobertaForSequenceClassification
			
    #------------albert-------------
    from transformers import AlbertTokenizer, AlbertForSequenceClassification
    model_name = 'albert-base-v1'
    tokenizer_name = 'albert-base-v1'
    ModelTokenizer = AlbertTokenizer
    PreTrainedModel = AlbertForSequenceClassification

    multilabel_cls(data_path,split_path, PreTrainedModel, epochs, batch_size, max_length ,ModelTokenizer, tokenizer_name, model_name, use_data_loader)

if __name__ == '__main__':
    main()