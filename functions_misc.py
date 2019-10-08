import pandas as pd
from tkinter import Tk
import sys
from PyQt5.QtWidgets import (QFileDialog, QAbstractItemView, QListView,
                             QTreeView, QApplication, QDialog)
import numpy as np
from scipy.interpolate import interp1d


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


def get_iframe_times(imeta_data, shutter_channel, nbr_iframes=None):

    if not nbr_iframes:
        nbr_iframes = [imeta_data.shape[1]]
    # initialize a counter
    cnt = 0
    # initialize the stopping variable
    go_on = 1
    # initialize the shutter closing
    shutter_close = [0]
    shutter_open = []
    while go_on:
        # if it ran out of threshold crossings
        if np.argwhere(shutter_channel[shutter_close[cnt]:] > 2.5).shape[0] == 0:
            go_on = 0
        else:
            open_temp = np.argwhere(shutter_channel[shutter_close[cnt]:] > 2.5)
            if open_temp.shape[0] > 0:
                shutter_open.append(open_temp[0][0] + shutter_close[cnt])
            close_temp = np.argwhere(shutter_channel[shutter_open[cnt]:] < 2.5)
            if close_temp.shape[0] > 0:
                shutter_close.append(close_temp[0][0] + shutter_open[cnt])
        # update the counter
        cnt += 1

    # if there are different numbers of shutter openings than frames, kill it
    # TODO: check whether this is needed for our experiments
    assert len(shutter_open) == len(nbr_iframes), "The number of shutter openings does not match the eye data"

    # get the frame times
    # initialize the list
    iframe_times = []
    # get a counter for the starts
    segment_start = 0
    # for all the frames
    for ind in range(len(nbr_iframes)):
        iframe_times[segment_start:
                     np.sum(nbr_iframes[:(ind+1)])] = imeta_data[1, segment_start:np.sum(nbr_iframes[:(ind+1)])] + \
                                                      shutter_open[ind] - imeta_data[1, segment_start]
        # update the segment
        segment_start += np.sum(nbr_iframes[:(ind+1)])
    return iframe_times, shutter_open


def rolling_window(data_in, window_size, func, *args, **kwargs):
    """Perform func on a rolling window"""
    # allocate memory for the output
    data_out = np.zeros_like(data_in)
    # get the subtraction to get to the bottom of the window
    window_bottom = np.int(np.floor(window_size / 2))

    for count in range(data_in[:, window_size:-window_size].shape[1]):
        # # assemble the regression vector
        # idx = count + window_bottom
        # regression_vector = data_in[count:count+window_size]
        # # fit the linear model
        # linear_model = ols().fit(np.array(range(regression_vector.shape[0])).reshape(-1, 1), regression_vector)
        # data_out[idx] = linear_model.coef_[0]
        # call func on the window
        data_out[:, count + window_bottom] = func(data_in[:, count:count+window_size], args[0], kwargs['axis'])
    # fill in the edges
    # data_out = np.array(data_out)
    data_out[:, :window_size] = np.tile(data_out[:, window_size], [window_size, 1]).T
    data_out[:, -window_size:] = np.tile(data_out[:, -window_size], [window_size, 1]).T
    return data_out


def interp_trace(x_known, y_known, x_target):
    """Interpolate a trace by building an interpolant"""
    # filter the values so the interpolant is trained only on sorted x points (required by the function)
    sorted_frames = np.hstack((True, np.invert(x_known[1:] <= x_known[:-1])))
    x_known = x_known[sorted_frames]
    y_known = y_known[sorted_frames]
    # also remove any NaN frames
    notnan = ~np.isnan(y_known)
    x_known = x_known[notnan]
    y_known = y_known[notnan]
    # create the interpolant
    interpolant = interp1d(x_known, y_known, kind='cubic', bounds_error=False, fill_value=np.mean(y_known))
    return interpolant(x_target)
