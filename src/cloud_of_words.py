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


#    This function returns a list of words from a file
def get_words_from_file(file):
    data = list(chain.from_iterable(json.load(file)))
    words = []

    for row in data:
        if CHARACTER_FIELD not in row[HEAD_TEXT_FIELD]:
            continue
        subj = row[HEAD_TEXT_FIELD][CHARACTER_FIELD]
        text = re.sub(r'[.,!?]', '', row[TEXT_FIELD].lower())  # Remove punctuation
        if subj is not None and len(text) > 0:
            words.append(text.split(" "))

    return list(chain.from_iterable(words))


#    This function plots a cloud of words from a list of words
def plot_cloud(words, title):
    df = pd.DataFrame(words, columns=['CONTENT'])

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
    plt.title(title)
    plt.tight_layout(pad=0)

    plt.show()


#    This function walks on the directory and sub-directories
#    and plots a cloud of words for each sub-directory
def cloud_of_words(directory):
    for parent_dir, sub_dir, files in os.walk(directory):
        # walk on every sub_dir
        for current_dir in sub_dir:
            dir_path = os.path.join(parent_dir, current_dir)
            files = os.listdir(dir_path)
            total_words = []

            for file_name in files:
                print(f"File Name: {file_name}\n")

                with open(os.path.join(dir_path, file_name), "r") as f:
                    words = get_words_from_file(f)
                    total_words.append(words)

            total_words = list(chain.from_iterable(total_words))

            plot_cloud(total_words, current_dir)


"""
    Main function
"""
cloud_of_words(directory_path)
