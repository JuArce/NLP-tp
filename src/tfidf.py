import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

from utils import flatten, load_json_files, apply_balance

def balance_classes(df, method='upsampling', random_seed=42):
    if method == 'upsampling':
        max_class_count = df['author'].value_counts().max()
        df = df.groupby('author').apply(lambda x: x.sample(max_class_count, replace=True, random_state=random_seed)).reset_index(drop=True)
    elif method == 'downsampling':
        min_class_count = df['author'].value_counts().min()
        df = df.groupby('author').apply(lambda x: x.sample(min_class_count, random_state=random_seed)).reset_index(drop=True)
    return df

def train_test_split_data(X, y, test_size=0.1, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_logistic_regression(X_train, y_train, max_iter=10000):
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test):
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

    return accuracy, recall, precision, f1

def plot_confusion_matrix(y_test, y_pred, idx2authors):
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=idx2authors.values(), yticklabels=idx2authors.values(),
        title='Confusion Matrix',
        ylabel='True label',
        xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    plt.show()

def plot_roc_curve(y, y_test, y_score, idx2authors, authors2idx):
    # Binarize the output
    y = label_binarize(y, classes=[*range(len(authors2idx))])
    n_classes = y.shape[1]
    y_test = label_binarize(y_test, classes=[*range(len(authors2idx))])

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for all classes
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve for author %s (area = %0.2f)' % (idx2authors[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for all authors')
    plt.legend(loc="lower right")
    plt.show()

def tfidf(directory):
    df = load_json_files(directory)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Balance classes if necessary
    BALANCE_CLASSES = True
    BALANCED_METHOD = 'upsampling' # or 'downsampling'
    RANDOM_SEED = 42

    if BALANCE_CLASSES:
        df = apply_balance(df, method=BALANCED_METHOD, random_seed=RANDOM_SEED)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text'])

    authors2idx = {author: idx for idx, author in enumerate(df['author'].unique())}
    idx2authors = {idx: author for author, idx in authors2idx.items()}

    y = df['author'].map(authors2idx).values

    X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.1, random_state=42)

    model = train_logistic_regression(X_train, y_train)

    accuracy, recall, precision, f1 = evaluate_model(model, X_train, X_test, y_train, y_test)

    plot_confusion_matrix(y_test, model.predict(X_test), idx2authors)

    y_score = model.decision_function(X_test)

    plot_roc_curve(y, y_test, y_score, idx2authors, authors2idx)

"""
    Main function
"""
directory_path = "../dataset"

tfidf(directory_path)