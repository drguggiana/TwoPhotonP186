import numpy as np
from paths import master_path
import h5py
from os.path import basename, join
from os import listdir
from functions_misc import multifolder


# load the target folders (select the experiment folders containing the files
file_path = multifolder(master_path)

# flag if we want to make a small test file - MM 17 Oct 2019
make_test = 0
if make_test:
    print("Creating a reduced file for suite2p training.")

# for all the selected files
for folders in file_path:
    # get the names of the bin files
    bin_files = [join(folders, el) for el in listdir(folders) if el.endswith('.bin')]
    green_path = bin_files[0]
    red_path = bin_files[1]

    # define the output path
    if make_test:
        # for a shorter file for suite2p training, modify the outpath and number of processed frames - MM 17 Oct 2019
        out_path = join(folders, basename(bin_files[0][:-4]) + '_' + basename(bin_files[1][:-4]) + '_test.hdf5')
    else:
        out_path = join(folders, basename(bin_files[0][:-4]) + '_' + basename(bin_files[1][:-4]) + '.hdf5')

    # So you remember which file you just processed - MM 17 Oct 2019
    print(out_path)

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
        nbr_frames = np.round(file_size/(x_res*y_res*2)).astype('int32')
        # for a shorter file for suite2p training, modify the outpath and number of processed frames - MM 17 Oct 2019
        if make_test:
            out_path = join(folders, basename(bin_files[0][:-4]) + '_' + basename(bin_files[1][:-4]) + '_test.hdf5')
            nbr_frames = int(nbr_frames*0.2)
        # set the position of the red file also at the beginning of the frame
        r.seek(4, 0)
        # create the hdf5 dataset
        out = d.create_dataset('data', (nbr_frames*2, x_res, y_res), dtype='int16')
        # Set some variables for a progress bar - MM 17 Oct 2019
        point = nbr_frames // 100
        increment = nbr_frames // 20

        # for all the frames
        for index, frames in enumerate(range(nbr_frames)):

            # Provide a progress bar every 5% of total frames converted
            if index % (5*point) == 0:
                num_eqs = index // increment
                num_spaces = (nbr_frames-index)//increment
                percentage = index//point
                print("\r[" + "=" * num_eqs + " " * num_spaces + "] " + str(percentage) + "%")

            # load the data
            data_green = np.fromfile(g, 'int16', count=x_res * y_res)
            data_red = np.fromfile(r, 'int16', count=x_res * y_res)
            # reshape it
            data_green = np.reshape(data_green, (x_res, y_res))
            data_red = np.reshape(data_red, (x_res, y_res))
            # save in the hdf5
            out[index*2, :, :] = data_green
            out[(index * 2) + 1, :, :] = data_red
