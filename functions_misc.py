import pandas as pd


def print_full(x):
    """pretty print pandas data frame in full in the terminal, taken from
    https://stackoverflow.com/questions/25351968/how-to-display-full-non-truncated-
    dataframe-information-in-html-when-convertin, answer by Karl Adler"""
    pd.set_option('display.max_rows', len(x))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', -1)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')
