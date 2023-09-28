# Python program to generate WordCloud
import json
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
from itertools import chain
from wordcloud import WordCloud, STOPWORDS

TEXT_FIELD = "text"
HEAD_TEXT_FIELD = "head_text"
CHARACTER_FIELD = "speaker/title"
stopwords = set(STOPWORDS)

# Specify the directory path containing the folders and files
directory_path = "../dataset"

for parent_dir, sub_dir, files in os.walk(directory_path):
    # walk on every sub_dir
    for current_dir in sub_dir:
        dir_path = os.path.join(parent_dir, current_dir)
        files = os.listdir(dir_path)
        total_words = []

        for file_name in files:
            print(f"File Name: {file_name}\n")

            with open(os.path.join(dir_path, file_name), "r") as f:
                data = list(chain.from_iterable(json.load(f)))
                words = []

                for row in data:
                    if CHARACTER_FIELD not in row[HEAD_TEXT_FIELD]:
                        continue
                    subj = row[HEAD_TEXT_FIELD][CHARACTER_FIELD]
                    text = re.sub(r'[.,!?]', '', row[TEXT_FIELD].lower())  # Remove punctuation
                    if subj is not None and len(text) > 0:
                        words.append(text.split(" "))

                total_words.append(list(chain.from_iterable(words)))

        total_words = list(chain.from_iterable(total_words))
        df = pd.DataFrame(total_words, columns=['CONTENT'])

        comment_words = ''

        for val in df.CONTENT:
            # typecaste each val to string
            val = str(val)

            # split the value
            tokens = val.split()

            comment_words += " ".join(tokens) + " "

        wordcloud = WordCloud(width=800, height=800,
                              background_color='white',
                              stopwords=stopwords,
                              min_font_size=10).generate(comment_words)

        # plot the WordCloud image
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud)
        # plt.axis("off")
        plt.title(current_dir)
        plt.tight_layout(pad=0)

        plt.show()
