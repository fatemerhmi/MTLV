import pandas as pd
import numpy as np

df_train = pd.read_csv("data/news/train.csv")
df_test = pd.read_csv("data/news/test.csv")

df_train.replace(np.nan, "", inplace=True)
df_test.replace(np.nan, "", inplace=True)

df_train.loc[:,'text'] = df_train.apply(lambda row: row['title']+" "+ row['desc'], axis=1)
df_test.loc[:,'text'] = df_test.apply(lambda row: row['title']+" "+ row['desc'], axis=1)

train_txt = " ".join (df_train.text)
test_txt = " ".join (df_test.text)

SAVE_DIR = "data/news"

file1 = open(f"{SAVE_DIR}/train.txt","w")
file1.write(train_txt) 
file1.close()

file2 = open(f"{SAVE_DIR}/test.txt","w")
file2.write(test_txt) 
file2.close()

from transformers import BertTokenizerFast
from transformers import BertForMaskedLM

tokenizer = BertTokenizerFast.from_pretrained("model_weights/bert-base-uncased")
model = BertForMaskedLM.from_pretrained('model_weights/bert-base-uncased')

#building the training dataset
from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="data/news/train.txt",
    block_size=128,
)

#data collector to help us batch differetn samples of dataset together
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="model_weights/news_BERT",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model("./model_weights/news_BERT")