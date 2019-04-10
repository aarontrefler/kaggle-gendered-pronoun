"""Utility support"""
import datetime
import sys

import pandas as pd


proj_path = "/Users/aarontrefler_temp2/Documents/My_Documents/Kaggle/kaggle-gendered-pronoun/"

bert_dir = proj_path + "bert/"
data_raw_dir = proj_path + "data/raw/"
data_interim_dir = proj_path + "data/interim/"
data_clean_dir = proj_path + "data/clean/"
models_dir = proj_path + "models/"


def cols_to_front(df, front_cols):
    """Moves selected coumns to front of DataFrame"""
    cols = list(df)
    front_cols.reverse()
    for col in front_cols:
        cols.insert(0, cols.pop(cols.index(col)))

    return df.loc[:, cols]


def display_df(df, n=1, tail=False, title=None):
    """Custom display method for DataFrames"""
    if title:
        print(title + ':')
    display(df.head(n), df.tail(n), df.shape) if tail else display(df.head(n), df.shape)


def create_datestamp():
    """Return current datestamp formatted as YYYYMMDDHH"""
    now = datetime.datetime.now()
    return "{year}{month:02}{day:02}{hour:02}".format(
        year=now.year, month=now.month, day=now.day, hour=now.hour)
