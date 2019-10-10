import numpy as np
import functions_plotting as plot
import functions_misc as misc


def plot_2pdataset(data_in):
    """Plot the raw, trial-averaged Ca data contained in the file, all protocols"""
    # if a path was inserted, then the file
    if data_in is str:
        # file = r'J:\Drago Guggiana Nilo\Data\DG_180816_a\2018_10_03\2\preProcessed.npz'
        contents = np.load(data_in, allow_pickle=True)

        data = contents['data'].item()
        # metadata = contents['metadata'].item()
    else:
        data = data_in

    # initialize a list to store the figure handles
    fig_list = []
    # for all the protocols
    for protocol in data:
        # analyze the DG info
        ca_data = data[protocol]['data']

        # # get the number of cells
        # cell_num = ca_data.shape[0]
        # # get the number of reps
        # rep_num = ca_data.shape[2]
        # get the number of stimuli
        stim_num = ca_data.shape[3]
        # get the number of time points
        time_num = ca_data.shape[1]

        # plot trial averages with concatenated orientation
        trial_average = np.nanmean(ca_data, axis=2)
        cat_matrix = np.reshape(trial_average, (-1, time_num*stim_num), order='F')

        fig_list.append(plot.plot_image([misc.normalize_matrix(cat_matrix, axis=1)], ylabel='Traces', title=protocol))
        fig_list[-1].show()

    return fig_list

