
        # %% markdown
# #Text Classification using BERT
#
# The is a basic implementation of text classification pipeline using BERT. The BERT model used has been taken from [huggingface](https://huggingface.co/transformers/). The dataset used is a custom dataset with two classes (labelled as 0 and 1). It is publically available [here](https://raw.githubusercontent.com/prateekjoshi565/Fine-Tuning-BERT/master/spamdata_v2.csv).
# %% codecell
!pip install transformers
# %% codecell
import csv
import pickle
import pandas as pd
import numpy as np
train = pd.read_csv("spamdata_v2.csv")

# %% codecell
print(len(train))
train.head()
# %% codecell
num_classes = 2
# %% codecell
from sklearn.model_selection import train_test_split
train_split, val_split = train_test_split(train, test_size=.05)
# %% codecell
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# %% codecell
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df
        self.text = df.text.values
        self.labels = df.label.values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):

        tokenized_data = tokenizer.tokenize(self.text[idx])
        to_append = ["[CLS]"] + tokenized_data[:self.max_length - 2] + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(to_append)
        input_mask = [1] * len(input_ids)
        padding = [0] * (self.max_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(input_mask, dtype=torch.long)
        }
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.df)

train_dataset = Dataset(train_split.fillna(""), tokenizer)
val_dataset = Dataset(val_split.fillna(""), tokenizer)
# train_dataset = Dataset(train.fillna(""), tokenizer, is_train=True, label_map=label_map)
# test_dataset = Dataset(test.fillna(""), tokenizer, is_train=False)
# %% codecell
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=50,              # total number of training epochs
    per_device_train_batch_size=64, # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    warmup_steps=500,               # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,
    dataloader_num_workers=2,
    report_to="tensorboard",
    label_smoothing_factor=0.1,
    evaluation_strategy="steps",
    eval_steps=500, # Evaluation and Save happens every 500 steps
    save_total_limit=3, # Only last 5 models are saved. Older ones are deleted.
    load_best_model_at_end=True,   #best model is always saved
)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
model.classifier = torch.nn.Linear(768, num_classes)
model.num_labels = num_classes
# %% codecell
trainer = Trainer(
    model=model,                         # the instantiated 
