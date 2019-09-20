import numpy as np
from paths import master_path
import h5py
from os.path import basename, join
from os import listdir
from functions_misc import multifolder


# load the target folders
file_path = multifolder(master_path)

# for all the selected files
for folders in file_path:
    # get the names of the bin files
    bin_files = [join(folders, el) for el in listdir(folders) if el.endswith('.bin')]
    green_path = bin_files[0]
    red_path = bin_files[1]
    # define the output path
    out_path = join(folders, basename(bin_files[0][:-4]) + '_' + basename(bin_files[1][:-4]) + '.hdf5')

    # read them, reformat and write
    with open(green_path, 'rb') as g, open(red_path, 'rb') as r, h5py.File(out_path, 'w') as d:
        # get the file size
        g.seek(0, 2)
        file_size = g.tell()
        g.seek(0, 0)
        # read the frame dimensions
        x_res = np.fromfile(g, 'int16', count=1)[0].astype('int64')
        y_res = np.fromfile(g, 'int16', count=1)[0].astype('int64')
        # update file_size to the current position in the file
        file_size -= g.tell()
        # get the number of frames
        nbr_frames = np.round(file_size/(x_res*y_res*2)).astype('int16')
        # set the position of the red file also at the beginning of the frame
        r.seek(4, 0)
        # create the hdf5 dataset
        out = d.create_dataset('data', (nbr_frames*2, x_res, y_res), dtype='int16')

        # for all the frames
        for index, frames in enumerate(range(nbr_frames)):
            # load the data
            data_green = np.fromfile(g, 'int16', count=x_res * y_res)
            data_red = np.fromfile(r, 'int16', count=x_res * y_res)
            # reshape it
            data_green = np.reshape(data_green, (x_res, y_res))
            data_red = np.reshape(data_red, (x_res, y_res))
            # save in the hdf5
            out[index*2, :, :] = data_green
            out[(index * 2) + 1, :, :] = data_red


