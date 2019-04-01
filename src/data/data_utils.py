"""Utility functions for data processing"""

def compute_offset_no_spaces(text, offset):
    """Taken from public Kaggle kernal: Taming the BERT - a baseline"""
    count = 0
    for pos in range(offset):
        if text[pos] != " ": count +=1
    return count


def count_chars_no_special(text):
    """Taken from public Kaggle kernal: Taming the BERT - a baseline"""
    count = 0
    special_char_list = ["#"]
    for pos in range(len(text)):
        if text[pos] not in special_char_list: count +=1
    return count


def count_length_no_special(text):
    """Taken from public Kaggle kernal: Taming the BERT - a baseline"""
    count = 0
    special_char_list = ["#", " "]
    for pos in range(len(text)):
        if text[pos] not in special_char_list: count +=1
    return count
