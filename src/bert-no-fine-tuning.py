from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
#plot roc curve
from sklearn.metrics import roc_curve, auc
import pandas as pd
import torch
import os
import json

from utils import flatten, remove_punctuation, average, round_list

def load_json_files(directory):
    data = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                author = os.path.basename(root)
                with open(os.path.join(root, file)) as f:
                    dialogs = flatten(json.load(f))
                    for dialog in dialogs:
                        if dialog['head_type'] == 'speaker/title':
                            data.append({'author': author, 'text': dialog['text']})
    return pd.DataFrame(data)

def bert_no_fine_tuning(directory):
    df = load_json_files(directory)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Balance classes if necessary
    BALANCE_CLASSES = True
    BALANCED_METHOD = 'upsampling' #'upsampling' # or 'downsampling'
    RANDOM_SEED = 42

    if BALANCE_CLASSES:
        if BALANCED_METHOD == 'upsampling':
            max_class_count = df['author'].value_counts().max()
            df = df.groupby('author').apply(lambda x: x.sample(max_class_count, replace=True, random_state=RANDOM_SEED)).reset_index(drop=True)
        elif BALANCED_METHOD == 'downsampling':
            min_class_count = df['author'].value_counts().min()
            df = df.groupby('author').apply(lambda x: x.sample(min_class_count, random_state=RANDOM_SEED)).reset_index(drop=True)

    print(df.groupby('author').count())

    authors = df['author'].unique()

    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    _X = [tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt') for text in df['text']]

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, X,y):
            self.X = X
            self.y = y
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    X = []
    # model = model.cuda()
    # device = torch.device("cuda")

    print(torch.backends.mps.is_available()) #the MacOS is higher than 12.3+
    print(torch.backends.mps.is_built()) #MPS is activated
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    dataset = Dataset(_X, df['author'].values)

    for x in _X:
        print(len(X)/len(_X)*100,"                               ", end='\r')
        mask = x['attention_mask'].to(device)
        input_id = x['input_ids'].squeeze(1).to(device)
        with torch.no_grad():
            test = model(input_ids=input_id, attention_mask=mask,return_dict=False)
            X.append(test[1].cpu().numpy())
    
    #remove first dimension of each X
    X =[x[0] for x in X]

    y = df['author'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

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
    #plot ticks
    tick_marks = np.arange(len(authors))
    plt.xticks(tick_marks, authors, rotation=45)
    plt.yticks(tick_marks, authors)

    plt.show()

    n_classes = len(authors)

    classes = [i for i in range(n_classes)]

    y_pred_prob = model.predict_proba(X_test)
    y_true = y_test
    authors2idx = {author:idx for idx, author in enumerate(authors)}
    #covert author names to numbers
    y_true = [authors2idx[author] for author in y_true]
    y_pred_prob = [[prob for prob in author] for author in y_pred_prob]


    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        _y_true = np.array(y_true) == i
        #_y_true = _y_true.astype(int)
        _y_pred_prob = np.array(y_pred_prob)[:, i]
        fpr[i], tpr[i], _ = roc_curve(_y_true, _y_pred_prob)
        roc_auc[i] = auc(fpr[i], tpr[i])


    plt.figure(figsize=(8, 8))
    lw = 2

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], "-", lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(authors[i], roc_auc[i]))

    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for Author Classification')
    plt.legend(loc="lower right")
    plt.show()



directory_path = "../dataset"
bert_no_fine_tuning(directory_path)