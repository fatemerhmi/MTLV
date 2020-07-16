from copy import deepcopy
import pandas as pd
from prettytable import PrettyTable


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

def split_dataset(data_path):

    #read the data
    df = pd.read_csv(data_path)

    #labels
    cols = df.columns
    label_cols = list(cols[6:])
    num_labels = len(label_cols)
    # print('Label columns: ', label_cols)
    
    #preprocess the data
    df_cls = preprocess_dataset(df)
    
    #shuffle 
    df_cls = df_cls.sample(frac=1).reset_index(drop=True)

    #splitting the data
    stratified_data_ids, stratified_data =  stratify(data=df_cls['labels'].values, classes=[0,1], ratios=[0.6,0.2,0.2], one_hot=False)
    train_df = df_cls.iloc[stratified_data_ids[0],:]
    test_df = df_cls.iloc[stratified_data_ids[1],:]
    val_df = df_cls.iloc[stratified_data_ids[2],:]

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    print('Train: ', len(train_df))
    print('Test: ', len(test_df))
    print('Val: ', len(val_df))

    label_counts_total = np.array(df_cls.labels.to_list()).sum(axis=0)
    label_counts_train = np.array(train_df.labels.to_list()).sum(axis=0)
    label_counts_test = np.array(test_df.labels.to_list()).sum(axis=0)
    label_counts_val = np.array(val_df.labels.to_list()).sum(axis=0)

    
    pretty=PrettyTable()
    pretty.field_names = ['Pathology', 'total', 'train', 'test','val']
    for pathology, cnt_total, cnt_train, cnt_test, cnt_val in zip(label_cols,label_counts_total, label_counts_train, label_counts_test, label_counts_val):
        pretty.add_row([pathology, cnt_total, cnt_train, cnt_test, cnt_val])
    print(pretty)
    
    return train_df, test_df, val_df, num_labels, label_cols

def save_dataset(df, file_name):
    df.to_csv(f'data/OpenI/cheXpertLabels/{file_name}.csv', index=False)

def main():    
    data_path = 'data/OpenI/OpenI_cheXpertLabels.csv'
    train_df, test_df, val_df=split(data_path)
    save_dataset(train_df,"train")
    save_dataset(test_df,"test")
    save_dataset(val_df,"val")


if __name__ == '__main__':
    main()