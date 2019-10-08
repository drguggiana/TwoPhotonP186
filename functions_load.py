import numpy as np


def load_lvd(file_name, header_only=False):
    """Read lvd file contents from a target file"""
    # read the header
    with open(file_name, 'rb') as f:
        header = np.fromfile(f, dtype='>d', count=4)
    # save the header contents
    scan_rate = header[0]
    num_channels = header[1]
    timestamp = header[2]
    input_range = header[3]

    # if the header only flag is on, skip this step
    if not header_only:
        # read the rest of the data
        with open(file_name, 'rb') as f:
            data = np.reshape(np.fromfile(f, dtype='>d')[4:], (-1, int(num_channels)))
    else:
        data = []

    return data, scan_rate, num_channels, timestamp, input_range


def load_eye_monitor_data(path, load_image=False, file_type=None):
    """load data from the eye monitor files"""
    # define the size of the metadata info based on the file type
    if file_type == "new_eye2":
        meta_info_size = 13
    else:
        meta_info_size = 9
    # read the header
    with open(path, 'rb') as f:
        # get the size of the file
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0, 0)
        # get the header
        meta_info = np.fromfile(f, dtype='>d', count=meta_info_size)
        # reset the position
        f.seek(0, 0)
        # get the size of the images
        if (meta_info[2] == 0) or np.abs(np.remainder(meta_info[2], 1) - 0) > 1e-10:
            size_x = meta_info[4].astype('uint64')
            size_y = meta_info[5].astype('uint64')
        else:
            # # if video was not taken
            # if file_type == 'old_eye2':
            #     size_x = (meta_info[4] - meta_info[2]).astype('uint64')
            #     size_y = (meta_info[5] - meta_info[3]).astype('uint64')
            # else:
            size_x = 0
            size_y = 0

        # select action depending on experiment type
        if file_type == 'new_eye2':
            # TODO: check this with one of the stage files
            # fix gaps on tracking

            # get the data
            imeta_temp = np.fromfile(f, dtype='>d')
            # rely on the ms timer to count frames
            nbr_frames = np.sum(imeta_temp > 10000)
            # find the position of those frames
            ms_timer = np.argwhere(imeta_temp > 10000)
            # find the short frames
            short_frames = np.argwhere(np.diff(ms_timer) < meta_info_size)
            # for all the short frames
            for frames in (short_frames - 1):
                # insert a vector with 4 zeros (nothing tracked) at the corresponding position
                imeta_temp = np.concatenate((imeta_temp[:frames*meta_info_size + 2], np.array([0, 0, 0, 0]),
                                             imeta_temp[frames*meta_info_size + 2:]))

            # reshape the matrix and output
            imeta_info = np.reshape(imeta_temp, (meta_info_size, nbr_frames))
        else:
            # get the number of frames (the 8 comes from the bytes)
            nbr_frames = np.uint64(file_size / (size_x * size_y + meta_info_size * 8))
            # if video is to be loaded, allocate memory to store it
            if load_image:
                idata = np.zeros((size_x, size_y, nbr_frames), dtype='>uint8')
            else:
                idata = np.array([])

            # allocate memory for the actual data
            imeta_info = np.zeros((meta_info_size, nbr_frames))

            # for all the frames
            for index, frames in enumerate(range(nbr_frames)):
                imeta_info[:, index] = np.fromfile(f, dtype='>d', count=meta_info_size)
                if load_image:
                    idata[:, :, index] = np.reshape(np.fromfile(f, dtype='>uint8', count=size_x*size_y), size_x, size_y)
                else:
                    f.seek(size_x*size_y, 1)
    return idata, imeta_info
