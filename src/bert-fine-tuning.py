import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt

from models import BertClassifier, Dataset, init_defaults
from utils import load_json_files

BATCH_SIZE = 1
EPOCHS = 5
LEARNING_RATE = 1e-6
BALANCE_CLASSES = True
BALANCED_METHOD = 'downsampling' # 'upsampling' or 'downsampling'
BASE_MODEL = 'bert-base-uncased'
RANDOM_SEED = 42
DATASET = "paragraphs" # "paragraphs" or "sentences"

directory_path = "../dataset"
df = load_json_files(directory_path)

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)



#select only X text per author
#df = df.groupby('Author').head(20).reset_index(drop=True)



TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_ID = f'{BASE_MODEL}-{DATASET}-epochs{EPOCHS}-lr{LEARNING_RATE:.2e}-batch{BATCH_SIZE}-{"" if BALANCE_CLASSES else "un"}balanced-{BALANCED_METHOD if BALANCE_CLASSES else ""}-seed{RANDOM_SEED}-{TIMESTAMP}'
print(RUN_ID)
MODEL_DIR = f'../models/{RUN_ID}'

if BALANCE_CLASSES:
    if BALANCED_METHOD == 'upsampling':
        max_class_count = df['author'].value_counts().max()
        df = df.groupby('author').apply(lambda x: x.sample(max_class_count, replace=True, random_state=RANDOM_SEED)).reset_index(drop=True)
    elif BALANCED_METHOD == 'downsampling':
        min_class_count = df['author'].value_counts().min()
        df = df.groupby('author').apply(lambda x: x.sample(min_class_count, random_state=RANDOM_SEED)).reset_index(drop=True)

#create dir RUN_ID
import os
os.makedirs(MODEL_DIR, exist_ok=True)

#save df
df.to_csv(f'{MODEL_DIR}/df.csv', index=False)
#count text per author
df.groupby('author').count()    


def train(model, train_data, val_data, learning_rate, epochs):

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    device = None

    if use_cuda or use_mps:
            if use_mps:
                device = torch.device('mps' if use_mps else 'cpu')
                model.to(device)
                # criterion = criterion.cuda()
            if use_cuda:
                device = torch.device("cuda" if use_cuda else "cpu")
                model = model.cuda()
                criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                
                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            torch.save(model.state_dict(), f'{MODEL_DIR}/model_{epoch_num}.pt')
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} | Train Accuracy: {total_acc_train / len(train_data): .3f} | Val Loss: {total_loss_val / len(val_data): .3f} | Val Accuracy: {total_acc_val / len(val_data): .3f}')
                  
def evaluate(model, test_data):

    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)

    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()

    device = None    

    if use_cuda or use_mps:
            if use_mps:
                model.to(device)
                device = torch.device('mps' if use_mps else 'cpu')
            if use_cuda:
                device = torch.device("cuda" if use_cuda else "cpu")
                model = model.cuda()

    total_acc_test = 0
    y_pred = []
    y_pred_prob = []
    y_true = []
    with torch.no_grad():

        for test_input, test_label in test_dataloader:

              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)

              output = model(input_id, mask)
              acc = (output.argmax(dim=1) == test_label).sum().item()
              total_acc_test += acc
              y_pred.append(output.argmax(dim=1).item())
              y_pred_prob.append(output.softmax(dim=1).squeeze(0).cpu().numpy())
              y_true.append(test_label.item())
    
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
    return y_pred, y_pred_prob, y_true

np.random.seed(RANDOM_SEED)
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=RANDOM_SEED), 
                                     [int(.8*len(df)), int(.9*len(df))])

print(len(df_train),len(df_val), len(df_test))

#save train, val, test
df_train.to_csv(f'{MODEL_DIR}/train.csv', index=False)
df_val.to_csv(f'{MODEL_DIR}/val.csv', index=False)
df_test.to_csv(f'{MODEL_DIR}/test.csv', index=False)

tokenizer, authors, author2idx, idx2author, model = init_defaults(df, BASE_MODEL)
EPOCHS = EPOCHS
model = BertClassifier()
LR = LEARNING_RATE
              
train(model, df_train, df_val, LR, EPOCHS)

y_pred, y_pred_prob, y_true = evaluate(model, df_test)


# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
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

# _, y_pred_prob, y_true = eval(model, df_test)


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    _y_true = np.array(y_true) == i
    _y_true = _y_true.astype(int)
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

print("Test Metrics")
print('Accuracy: {:.2f}'.format(accuracy_score(y_true, y_pred)))
print('Precision: {:.2f}'.format(precision_score(y_true, y_pred, average='macro')))
print('Recall: {:.2f}'.format(recall_score(y_true, y_pred, average='macro')))
print('F1: {:.2f}'.format(f1_score(y_true, y_pred, average='macro')))

# y_pred, y_pred_prob, y_true = eval(model, df_train)

print("Train Metrics")
print('Accuracy: {:.2f}'.format(accuracy_score(y_true, y_pred)))
print('Precision: {:.2f}'.format(precision_score(y_true, y_pred, average='macro')))
print('Recall: {:.2f}'.format(recall_score(y_true, y_pred, average='macro')))
print('F1: {:.2f}'.format(f1_score(y_true, y_pred, average='macro')))