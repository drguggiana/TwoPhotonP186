from paths import master_path
from functions_misc import tk_killwindow
from tkinter.filedialog import askopenfilenames
import numpy as np
from functions_gratings import calculate_dsi_osi, project_angles
import matplotlib.pyplot as plt


tk_killwindow()

# get the files
filenames = askopenfilenames(initialdir=master_path)

# for all the filenames
for files in filenames:
    # load the data
    contents = np.load(files, allow_pickle=True)
    data = contents['data'].item()
    # check if there is a DG field, otherwise skip it
    if 'DG' not in data:
        continue
    # calculate DSI, OSI for each ROI
    data = data['DG']
    # get the number of ROIs
    roi_number = data['data'].shape[0]
    # calculate the trial average
    trial_average = np.nanmean(data['data'], axis=2)
    # get the angle exponentials
    angle_exponentials = project_angles(np.unique(data['trial_info'][:, 1]).astype(np.int32))
    # allocate memory for the indexes
    dsi_osi_list = [calculate_dsi_osi(trial_average[roi, :, :], angle_exponentials) for roi in np.arange(roi_number)]

    # plot the dsi and osi of the population
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(np.array([el[0] for el in dsi_osi_list]))
    plt.show()

# TODO: add clustering
# TODO: add pixel maps
# TODO: add ODI calculation
# TODO: add responsive vs non-responsive
# TODO: add temporal and spatial frequencies

print('yay')