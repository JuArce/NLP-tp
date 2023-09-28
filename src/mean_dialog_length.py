import json
import os
from itertools import chain

from matplotlib import pyplot as plt

from utils import flatten, remove_punctuation, average, round_list

TEXT_FIELD = "text"
HEAD_TEXT_FIELD = "head_text"
CHARACTER_FIELD = "speaker/title"


def get_dialogs_from_file(file):
    data = list(chain.from_iterable(json.load(file)))
    dialogs = []

    for row in data:
        if CHARACTER_FIELD not in row[HEAD_TEXT_FIELD]:
            continue
        subj = row[HEAD_TEXT_FIELD][CHARACTER_FIELD]
        text = remove_punctuation(row[TEXT_FIELD].lower())  # Remove punctuation
        if subj is not None and len(text) > 0:
            dialogs.append(text)

    return dialogs


def get_dialogs_length(dialogs):
    # Return the length of each dialog
    return [len(dialog) for dialog in dialogs]


def plot_dialogs_mean_by_writer(dialogs_length_per_writer):
    names = list(dialogs_length_per_writer.keys())
    values = list(dialogs_length_per_writer.values())

    # Round all values to integers
    values = round_list(values, 0)

    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
    bars = plt.bar(range(len(dialogs_length_per_writer)), values, tick_label=names)

    # Rotate the tick labels vertically and increase the space for them
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.3)  # Adjust the bottom margin as needed

    # Annotate each bar with its rounded value
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha='center')

    plt.title("Mean dialog length by writer")
    plt.show()


def mean_dialog_length(directory):
    dialogs_length_per_writer = {}
    for parent_dir, sub_dirs, files in os.walk(directory):
        for current_dir in sub_dirs:
            dir_path = os.path.join(parent_dir, current_dir)
            files = os.listdir(dir_path)
            total_dialogs_length = []

            for file_name in files:
                file_path = os.path.join(dir_path, file_name)
                with open(file_path, 'r') as file:
                    dialogs = get_dialogs_from_file(file)
                    dialogs_length = get_dialogs_length(dialogs)
                    total_dialogs_length.append(dialogs_length)

            # Flatten total_dialogs_length
            total_dialogs_length = flatten(total_dialogs_length)
            # Calculate the mean of the dialogs length
            avg = average(total_dialogs_length)
            dialogs_length_per_writer.setdefault(current_dir, avg)

    plot_dialogs_mean_by_writer(dialogs_length_per_writer)


"""
    Main function
"""
directory_path = "../dataset"

mean_dialog_length(directory_path)
