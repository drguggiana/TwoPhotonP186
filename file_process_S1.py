import numpy as np
import functions_plotting as plot
import functions_misc as misc

file = r'J:\Drago Guggiana Nilo\Data\DG_180816_a\2018_10_03\2\preProcessed.npz'
contents = np.load(file, allow_pickle=True)

data = contents['data'].item()
metadata = contents['metadata'].item()

# analyze the DG info
DG_data = data['DG']['data']

# get the number of cells
cell_num = DG_data.shape[0]
# get the number of reps
rep_num = DG_data.shape[2]
# get the number of stimuli
stim_num = DG_data.shape[3]
# get the number of time points
time_num = DG_data.shape[1]

# plot trial averages with concatenated orientation
trial_average = np.nanmean(DG_data, axis=2)
cat_matrix = np.reshape(trial_average, (-1, time_num*stim_num), order='F')
# cat_matrix = trial_average[:, :, 0]

image = plot.plot_image([misc.normalize_matrix(cat_matrix, axis=1)])
image.show()
print('yay')
