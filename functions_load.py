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


def load_eye_monitor_data_eye1UD(path, load_image=False):
    """load data from the eye monitor files"""
    # define the size of the metadata info
    meta_info_size = 9
    # read the header
    with open(path, 'rb') as f:
        # get the size of the file
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0, 0)
        # get the header
        meta_info = np.fromfile(f, dtype='uint8', count=meta_info_size)

        # get the size of the images
        if (meta_info[2] == 0) or (np.remainder(meta_info[2], 1) is not 0):
            size_x = meta_info[4]
            size_y = meta_info[5]
        else:
            # if video was not taken
            size_x = 0
            size_y = 0

        # get the number of frames (the 8 comes from the bytes)
        nbr_frames = file_size/(size_x*size_y + meta_info_size*8)

        # video is to be loaded, allocate memory to store it
        if load_image:
            idata = np.zeros((size_x, size_y, nbr_frames), dtype='uint8')
        else:
            idata = np.array([])

        # allocate memory for the actual data
        imeta_info = np.zeros((meta_info_size, nbr_frames))

        # for all the frames
        for index, frames in enumerate(range(nbr_frames)):
            imeta_info[:, index] = np.fromfile(f, dtype='uint8', count=meta_info_size)
            if load_image:
                idata[:, :, index] = np.reshape(np.fromfile(f, dtype='uint8', count=size_x*size_y), size_x, size_y)
            else:
                f.seek(size_x*size_y, 1)
    return idata, imeta_info
