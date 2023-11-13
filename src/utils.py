import os
import json
import re
from itertools import chain
import pandas as pd

def flatten(matrix):
    return list(chain.from_iterable(matrix))


def remove_punctuation(text):
    return re.sub(r'[.,!?]', '', text)


def average(numbers):
    return sum(numbers) / len(numbers)


def round_list(numbers, digits):
    return [round(number, digits) for number in numbers]

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

def apply_balance(BALANCED_METHOD, RANDOM_SEED, df):
    if BALANCED_METHOD == 'upsampling':
        max_class_count = df['author'].value_counts().max()
        df = df.groupby('author').apply(lambda x: x.sample(max_class_count, replace=True, random_state=RANDOM_SEED)).reset_index(drop=True)
    elif BALANCED_METHOD == 'downsampling':
        min_class_count = df['author'].value_counts().min()
        df = df.groupby('author').apply(lambda x: x.sample(min_class_count, random_state=RANDOM_SEED)).reset_index(drop=True)
    return df