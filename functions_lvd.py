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
