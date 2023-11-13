import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

from utils import apply_balance, load_json_files


def balance_classes(df, balance_classes, balanced_method, random_seed):
    if balance_classes:
        df = apply_balance(balanced_method, random_seed, df)
    return df


def tokenize_texts(df):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    _X = [tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt') for text in df['text']]
    return _X


def create_dataset(X, y):
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    return Dataset(X, y)


def get_device():
    if torch.backends.mps.is_available():
        print(torch.backends.mps.is_available())
        print(torch.backends.mps.is_built())
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_embeddings(model, X, device):
    embeddings = []
    for x in X:
        mask = x['attention_mask'].to(device)
        input_id = x['input_ids'].squeeze(1).to(device)
        with torch.no_grad():
            test = model(input_ids=input_id, attention_mask=mask, return_dict=False)
            embeddings.append(test[1].cpu().numpy())
    return [x[0] for x in embeddings]


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, authors):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print("Test metrics")
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("Precision:", precision)
    print("F1:", f1)
    print("-------------------------------------")
    print("Train metrics")
    print("Accuracy:", accuracy_score(y_train, model.predict(X_train)))
    print("Recall:", recall_score(y_train, model.predict(X_train), average='macro'))
    print("Precision:", precision_score(y_train, model.predict(X_train), average='macro'))
    print("F1:", f1_score(y_train, model.predict(X_train), average='macro'))

    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plot ticks
    tick_marks = np.arange(len(authors))
    plt.xticks(tick_marks, authors, rotation=45)
    plt.yticks(tick_marks, authors)

    plt.show()

    n_classes = len(authors)

    classes = [i for i in range(n_classes)]

    y_pred_prob = model.predict_proba(X_test)
    y_true = y_test

    authors2idx = {author: idx for idx, author in enumerate(authors)}
    # covert author names to numbers
    y_true = [authors2idx[author] for author in y_true]
    y_pred_prob = [[prob for prob in author] for author in y_pred_prob]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        _y_true = np.array(y_true) == i
        # _y_true = _y_true.astype(int)
        _y_pred_prob = np.array(y_pred_prob)[:, i]
        fpr[i], tpr[i], _ = roc_curve(_y_true, _y_pred_prob)
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 8))
    lw = 2

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], "-", lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(authors[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for all authors')
    plt.legend(loc="lower right")
    plt.show()


def bert_no_fine_tuning(directory, balance_classes=True, balanced_method='upsampling', random_seed=42):
    df = load_json_files(directory)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = balance_classes(df, balance_classes, balanced_method, random_seed)

    authors = df['author'].unique()

    _X = tokenize_texts(df)

    device = get_device()

    model = BertModel.from_pretrained('bert-base-uncased')
    model.to(device)

    X = get_embeddings(model, _X, device)

    y = df['author'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test, authors)


directory_path = "../dataset"
bert_no_fine_tuning(directory_path)