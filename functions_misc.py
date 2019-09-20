import pandas as pd
from tkinter import Tk
import sys
from PyQt5.QtWidgets import (QFileDialog, QAbstractItemView, QListView,
                             QTreeView, QApplication, QDialog)


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


def tk_killwindow():
    """Prevents the appearance of the tk main window when using other GUI components"""
    # Create Tk root
    root = Tk()
    # Hide the main window
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    return None


class GetExistingDirectories(QFileDialog):
    def __init__(self, *args):
        super(GetExistingDirectories, self).__init__(*args)
        self.setOption(self.DontUseNativeDialog, True)
        self.setFileMode(self.Directory)
        self.setOption(self.ShowDirsOnly, True)
        self.findChildren(QListView)[0].setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.findChildren(QTreeView)[0].setSelectionMode(QAbstractItemView.ExtendedSelection)


def multifolder(path):
    """Using the above GetExistingDirectories class, show a GUI to select multiple folders, both taken from
    https://www.qtcentre.org/threads/34226-QFileDialog-select-multiple-directories?p=158482#post158482"""
    qapp = QApplication(sys.argv)
    dlg = GetExistingDirectories()
    dlg.setDirectory(path)
    file_path = []
    if dlg.exec_() == QDialog.Accepted:
        file_path = dlg.selectedFiles()
    return file_path
