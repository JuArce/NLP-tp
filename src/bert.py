import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset

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

def bert(directory):
    df = load_json_files(directory)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Balance classes if necessary
    BALANCE_CLASSES = True
    BALANCED_METHOD = 'downsampling' #'upsampling' # or 'downsampling'
    RANDOM_SEED = 42

    if BALANCE_CLASSES:
        if BALANCED_METHOD == 'upsampling':
            max_class_count = df['author'].value_counts().max()
            df = df.groupby('author').apply(lambda x: x.sample(max_class_count, replace=True, random_state=RANDOM_SEED)).reset_index(drop=True)
        elif BALANCED_METHOD == 'downsampling':
            min_class_count = df['author'].value_counts().min()
            df = df.groupby('author').apply(lambda x: x.sample(min_class_count, random_state=RANDOM_SEED)).reset_index(drop=True)

    print(df.groupby('author').count())

    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['author'].unique()))

    # Tokenize and prepare data
    tokenized_texts = [tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt') for text in df['text']]
    input_ids = torch.cat([tokenized_text['input_ids'] for tokenized_text in tokenized_texts], dim=0)
    attention_masks = torch.cat([tokenized_text['attention_mask'] for tokenized_text in tokenized_texts], dim=0)
    labels = torch.tensor(df['author'].map({author: idx for idx, author in enumerate(df['author'].unique())}).values)


    # Split the data into training and testing sets
    input_ids_train, input_ids_test, attention_masks_train, attention_masks_test, labels_train, labels_test = train_test_split(
        input_ids, attention_masks, labels, test_size=0.1, random_state=42
    )

    # Create DataLoader for training and testing sets
    train_dataset = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    test_dataset = TensorDataset(input_ids_test, attention_masks_test, labels_test)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Training
    print(torch.backends.mps.is_available()) #the MacOS is higher than 12.3+
    print(torch.backends.mps.is_built()) #MPS is activated
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    for epoch in range(3):  # You can adjust the number of epochs
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids_batch, attention_masks_batch, labels_batch = tuple(t.to(device) for t in batch)

            optimizer.zero_grad()
            outputs = model(input_ids_batch, attention_mask=attention_masks_batch, labels=labels_batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss}")

    # Evaluation
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids_batch, attention_masks_batch, labels_batch = tuple(t.to(device) for t in batch)
            outputs = model(input_ids_batch, attention_mask=attention_masks_batch)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_batch.cpu().numpy())

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=df['author'].unique(), yticklabels=df['author'].unique(),
        title='Confusion Matrix',
        ylabel='True label',
        xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    plt.show()

    # Binarize the output
    n_classes = len(df['author'].unique())
    y = label_binarize(all_labels, classes=[*range(n_classes)])
    y_score = torch.softmax(torch.tensor(all_preds), dim=1).numpy()

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    for i in range(n_classes):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic for author ' + df['author'].unique()[i])
        plt.legend(loc="lower right")
        plt.show()

# Main function
directory_path = "../dataset"
bert(directory_path)
