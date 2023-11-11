from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch
import numpy as np




def init_defaults(df, _model):
    global tokenizer, authors, author2idx, idx2author, model
    tokenizer = BertTokenizer.from_pretrained(_model)
    authors = df['author'].unique()
    author2idx = {author: idx for idx, author in enumerate(authors)}
    idx2author = {idx: author for idx, author in enumerate(authors)}
    model = _model
    return tokenizer, authors, author2idx, idx2author, model

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        
        self.labels = [author2idx[author] for author in df['author']]
        self.texts = [ tokenizer(text, padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

#https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f
class BertClassifier(nn.Module):

    def __init__(self):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(model)
        self.linear = nn.Linear(768, len(authors))
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        linear_output = self.linear(pooled_output)
        final_layer = self.relu(linear_output)
        return final_layer