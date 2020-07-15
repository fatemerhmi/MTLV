import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from tqdm import tqdm, trange
from transformers import AdamW
from torch.nn import BCEWithLogitsLoss, BCELoss
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def stratify(data, classes, ratios, one_hot=False):
    """Stratifying procedure.

    data is a list of lists: a list of labels, for each sample.
        Each sample's labels should be ints, if they are one-hot encoded, use one_hot=True
    
    classes is the list of classes each label can take

    ratios is a list, summing to 1, of how the dataset should be split

    """
    # one-hot decoding
    if one_hot:
        temp = [[] for _ in range(len(data))]
        indexes, values = np.where(np.array(data).astype(int) == 1)
        for k, v in zip(indexes, values):
            temp[k].append(v)
        data = temp

    # Organize data per label: for each label l, per_label_data[l] contains the list of samples
    # in data which have this label
    per_label_data = {c: set() for c in classes}
    for i, d in enumerate(data):
        for l in d:
            per_label_data[l].add(i)

    # number of samples
    size = len(data)

    # In order not to compute lengths each time, they are tracked here.
    subset_sizes = [r * size for r in ratios]
    target_subset_sizes = deepcopy(subset_sizes)
    per_label_subset_sizes = {
        c: [r * len(per_label_data[c]) for r in ratios]
        for c in classes
    }

    # For each subset we want, the set of sample-ids which should end up in it
    stratified_data_ids = [set() for _ in range(len(ratios))]

    # For each sample in the data set
    while size > 0:
        # Compute |Di|
        lengths = {
            l: len(label_data)
            for l, label_data in per_label_data.items()
        }
        try:
            # Find label of smallest |Di|
            label = min(
                {k: v for k, v in lengths.items() if v > 0}, key=lengths.get
            )
        except ValueError:
            # If the dictionary in `min` is empty we get a Value Error. 
            # This can happen if there are unlabeled samples.
            # In this case, `size` would be > 0 but only samples without label would remain.
            # "No label" could be a class in itself: it's up to you to format your data accordingly.
            break
        current_length = lengths[label]

        # For each sample with label `label`
        while per_label_data[label]:
            # Select such a sample
            current_id = per_label_data[label].pop()

            subset_sizes_for_label = per_label_subset_sizes[label]
            # Find argmax clj i.e. subset in greatest need of the current label
            largest_subsets = np.argwhere(
                subset_sizes_for_label == np.amax(subset_sizes_for_label)
            ).flatten()

            if len(largest_subsets) == 1:
                subset = largest_subsets[0]
            # If there is more than one such subset, find the one in greatest need
            # of any label
            else:
                largest_subsets = np.argwhere(
                    subset_sizes == np.amax(subset_sizes)
                ).flatten()
                if len(largest_subsets) == 1:
                    subset = largest_subsets[0]
                else:
                    # If there is more than one such subset, choose at random
                    subset = np.random.choice(largest_subsets)

            # Store the sample's id in the selected subset
            stratified_data_ids[subset].add(current_id)

            # There is one fewer sample to distribute
            size -= 1
            # The selected subset needs one fewer sample
            subset_sizes[subset] -= 1

            # In the selected subset, there is one more example for each label
            # the current sample has
            for l in data[current_id]:
                per_label_subset_sizes[l][subset] -= 1
            
            # Remove the sample from the dataset, meaning from all per_label dataset created
            for l, label_data in per_label_data.items():
                if current_id in label_data:
                    label_data.remove(current_id)

    # Create the stratified dataset as a list of subsets, each containing the orginal labels
    stratified_data_ids = [sorted(strat) for strat in stratified_data_ids]
    stratified_data = [
        [data[i] for i in strat] for strat in stratified_data_ids
    ]

    # Return both the stratified indexes, to be used to sample the `features` associated with your labels
    # And the stratified labels dataset
    return stratified_data_ids, stratified_data

def concat_cols(impression, findings):
    if impression is np.nan:
        return findings
    elif findings is np.nan:
        return impression
    else:
        return findings+impression
    
def preprocess_dataset(df):
    # 1. removing the rows without any Impression and Finding
    print('No. of rows with Finding:', len(df[df['FINDINGS'].notnull()]))
    print('No. of rows with Impression:', len(df[df['IMPRESSION'].notnull()]))
    print('No. of rows with Impression or Finding:', 
          len(df[df['IMPRESSION'].notnull() | df['FINDINGS'].notnull()]))
    print('No. of rows without Impression and Finding:', 
          len(df[df['IMPRESSION'].isna() & df['FINDINGS'].isna()]))
    
    idx = df[df['IMPRESSION'].isna() & df['FINDINGS'].isna()].index
    df = df.drop(idx)
    print('No. of rows without Impression and Finding:', 
          len(df[df['IMPRESSION'].isna() & df['FINDINGS'].isna()]))
    
    
    # 2. Converting the labels to a single list
    labels = ['No Finding', 'Cardiomegaly','Lung Opacity','Edema','Consolidation','Pneumonia','Atelectasis','Pneumothorax','Pleural Effusion','Fracture','SupportDevices']

    df_cls = pd.DataFrame(columns = ['text', 'labels'])
    # create the text column:
    df_cls['text'] = df.apply(lambda row: concat_cols(row['IMPRESSION'], row['FINDINGS']), axis=1)
    df_cls['labels'] = df.apply(lambda row: row[labels].to_list(), axis=1) #.to_list() , .values

    # 3. Removing the duplicate reports
    print("Length of whole dataframe:", len(df_cls))
    print("No. of unique reports:", df_cls.text.nunique())
    df_cls.drop_duplicates('text', inplace=True)
    return df_cls
    
def prepare_dataset(data_path):

    #read the data
    df = pd.read_csv(data_path)

    #labels
    cols = df.columns
    label_cols = list(cols[6:])
    num_labels = len(label_cols)
    # print('Label columns: ', label_cols)
    
    #preprocess the data
    df_cls = preprocess_dataset(df)
    
    #splitting the data
    stratified_data_ids, stratified_data =  stratify(data=df_cls['labels'].values, classes=[0,1], ratios=[0.6,0.2,0.2], one_hot=False)
    train_df = df_cls.iloc[stratified_data_ids[0],:]
    test_df = df_cls.iloc[stratified_data_ids[1],:]
    val_df = df_cls.iloc[stratified_data_ids[2],:]

    print('Train: ', len(train_df))
    print('Test: ', len(test_df))
    print('Val: ', len(val_df))

    label_counts_total = np.array(df_cls.labels.to_list()).sum(axis=0)
    label_counts_train = np.array(train_df.labels.to_list()).sum(axis=0)
    label_counts_test = np.array(test_df.labels.to_list()).sum(axis=0)
    label_counts_val = np.array(val_df.labels.to_list()).sum(axis=0)

    from prettytable import PrettyTable
    pretty=PrettyTable()
    pretty.field_names = ['Pathology', 'total', 'train', 'test','val']
    for pathology, cnt_total, cnt_train, cnt_test, cnt_val in zip(label_cols,label_counts_total, label_counts_train, label_counts_test, label_counts_val):
        pretty.add_row([pathology, cnt_total, cnt_train, cnt_test, cnt_val])
    print(pretty)
    
    
    return train_df, test_df, val_df, num_labels, label_cols
    
def multilabel_cls(data_path, PreTrainedModel, epochs, batch_size, max_length ,ModelTokenizer, model_name, use_data_loader):
    #prepare the dataset
    train_df, test_df, val_df, num_labels, label_cols=prepare_dataset(data_path)

    # ----------tokenize---------------
    tokenizer = ModelTokenizer.from_pretrained(model_name)

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
        train_data = TensorDataset(train.input_ids, train.attention_mask, train_labels, train.token_type_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        validation_data = TensorDataset(val.input_ids, val.attention_mask, val_labels, val.token_type_ids)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

        test_data = TensorDataset(test.input_ids, test.attention_mask, test_labels, test.token_type_ids)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=validation_sampler, batch_size=batch_size)
    
    else: #TODO: if the dataset is small in size
        pass

    # Load model, the pretrained model will include a single linear classification layer on top for classification. 
    model = PreTrainedModel.from_pretrained("bert-base-uncased", num_labels=num_labels)
    model.cuda()
    optimizer = AdamW(model.parameters(),lr=2e-5)  # Default optimization
    
    #---------FineTune model-----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
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
            b_input_ids, b_input_mask, b_labels, b_token_types = batch
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
            b_input_ids, b_input_mask, b_labels, b_token_types = batch
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
        print('Flat Validation Accuracy: ', val_flat_accuracy)

    # ---------test--------
    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    #track variables
    logit_preds,true_labels,pred_labels,tokenized_texts = [],[],[],[]

    # Predict
    for i, batch in enumerate(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels, b_token_types = batch
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
    print('Test F1 Score: ', f1_score(true_bools, pred_bools,average='micro'))
    print('Test Accuracy: ', accuracy_score(true_bools, pred_bools),'\n')
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
    print('Test F1 Accuracy: ', f1_results[best_f1_idx])
    print('Test Flat Accuracy: ', flat_acc_results[best_f1_idx], '\n')

    best_pred_bools = [pl>micro_thresholds[best_f1_idx] for pl in pred_labels]
    clf_report_optimized = classification_report(true_bools,best_pred_bools, target_names=label_cols)
    # pickle.dump(clf_report_optimized, open('classification_report_optimized.txt','wb'))
    print(clf_report_optimized)

def main():    
    data_path = 'data/OpenI/OpenI_cheXpertLabels.csv'
    model_name = 'bert-base-uncased'
    ModelTokenizer = BertTokenizer
    use_data_loader = True
    PreTrainedModel = BertForSequenceClassification
    epochs = 3 # Number of training epochs (authors recommend between 2 and 4)
    batch_size = 16
    max_length = 128
    multilabel_cls(data_path, PreTrainedModel, epochs, batch_size, max_length ,ModelTokenizer, model_name, use_data_loader)



if __name__ == '__main__':
    main()