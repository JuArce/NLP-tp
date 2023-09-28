import re
from itertools import chain


def flatten(matrix):
    return list(chain.from_iterable(matrix))


def remove_punctuation(text):
    return re.sub(r'[.,!?]', '', text)


def average(numbers):
    return sum(numbers) / len(numbers)


def round_list(numbers, digits):
    return [round(number, digits) for number in numbers]
