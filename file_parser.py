# File parser
from os import walk, listdir, mkdir
from os.path import isfile, join, exists, basename
import pandas as pd
from functions_lvd import *
import datetime as dt
import tkinter
from tkinter import messagebox
from functions_misc import print_full
from shutil import move
from paths import *

# hide main window
root = tkinter.Tk()
root.withdraw()




# Parse aux files
# get the aux file list
aux_folders = [dirpath for dirpath, dirnames, filenames in walk(aux_path) if filenames]

# allocate memory to store the actual files
aux_struct = []
# for all the remaining files
for aux in aux_folders:

    # load the files in that folder
    folder_files = [el for el in listdir(aux) if el.endswith('.lvd')]
    # extract the animal and experiment
    aux_tuple = [[el[11:15], el[:11]] for el in folder_files]
    aux_experiment = [el[0] for el in aux_tuple]
    aux_animal = [el[1] for el in aux_tuple]

    # get the number of experiments
    num_exp = len(aux_experiment)

    # for all the experiments
    for experiment in range(num_exp):
        # allocate a sublist
        sublist = [aux_animal[experiment], aux_experiment[experiment], aux, folder_files[experiment],
                   folder_files[experiment][:-3] + 'eye1', folder_files[experiment][:-3] + 'eye2']

        # if there's a text file, add it
        if isfile(join(aux, folder_files[experiment][:-3] + 'txt')):
            sublist.append(folder_files[experiment][:-3] + 'txt')
        else:
            sublist.append([])

        # load the info into the main list
        aux_struct.append(sublist)

# turn the list into a dataframe
aux_struct = pd.DataFrame(aux_struct, columns=['animal', 'experiment', 'folder', 'lvd_file', 'eye1_file', 'eye2_file',
                          'txt_file'])

# allocate memory for the file timestamp
time_list = []
# load the timestamp from each file
for aux in range(len(aux_struct.index)):
    _, _, _, timestamp, _ = load_lvd(join(aux_struct.loc[aux, 'folder'], aux_struct.loc[aux, 'lvd_file']),
                                     header_only=True)
    # format the timestamp
    timestamp = str(timestamp)
    # time_list.append('_'.join([timestamp[:4], timestamp[4:6], timestamp[6:8], timestamp[8:10], timestamp[10:12],
    #                            timestamp[12:14]]))
    time_list.append(dt.datetime.strptime(timestamp, '%Y%m%d%H%M%S.%f'))

# add the timestamps to the data frame
aux_struct['timestamp'] = time_list

# put timestamp at the beginning
aux_struct = aux_struct[['timestamp', 'animal', 'experiment', 'folder', 'lvd_file', 'eye1_file', 'eye2_file',
                        'txt_file']]
aux_struct.sort_values('timestamp', inplace=True)
# set the timestamp as an index
aux_struct.reset_index(None, inplace=True, drop=True)

# get a list of the bin files in the directory
bin_files = [join(bin_path, el) for el in listdir(bin_path) if el.endswith('525.ini')]

# allocate a list for each file's row
path_struct = []

# for all the bin files
for files in bin_files:
    # allocate memory for the file contents
    ini_contents = []
    # read the file
    with open(files) as f:
        # skip the first line, which only says "main"
        f.readline()
        for lines in f:
            ini_contents.append(lines.split("\"")[1])
    # get the files date and time
    file_timestamp = dt.datetime.strptime(ini_contents[1][:-1], '%Y-%m-%d %H:%M:%S.%f')

    # match the corresponding aux file
    tar_aux_time = np.argmin(np.array(np.abs(aux_struct.loc[:, 'timestamp'] - file_timestamp)))
    # make sure the file is actually there
    assert np.array(np.abs(aux_struct.loc[tar_aux_time, 'timestamp'] - file_timestamp)).astype('timedelta64[s]') \
        < np.timedelta64(10800, 's'), "Binary file is more than 3 hours away from the aux file, probably wrong file"

    # Determine whether there is a stim file and load it if so

    # assemble the folder path for the stim files
    stim_subpath = join(stim_path, file_timestamp.strftime('%Y_%m_%d'))

    # if the path exists
    if exists(stim_subpath):
        # get the files in the stim path
        stim_files = [join(stim_subpath, el) for el in listdir(stim_subpath) if el.endswith('.mat')]
        # get their times
        stim_times = np.array([dt.datetime.strptime(basename(el)[9:28], '%Y_%m_%d_%H_%M_%S') for el in stim_files])
        # find the closest file within a tolerance
        tar_stim_time = np.argmin(np.array(np.abs(stim_times - file_timestamp)))
        try:
            assert np.array(stim_times[tar_stim_time] - file_timestamp).astype('timedelta64[s]') \
                < np.timedelta64(30, 's'), "File is too far away in the future"
            assert np.array(stim_times[tar_stim_time] - file_timestamp).astype('timedelta64[s]') \
                > np.timedelta64(-180, 's'), "File is too far away in the past"
            stim_file = 1
        except AssertionError as m:
            print(m.args)
            stim_file = 0
    else:
        stim_file = 0
        tar_stim_time = 0
        stim_files = []

    # fill in a list of lists with the paths to all the connected files
    # initialize the list
    temp_struct = 12*[None]
    # add the animal
    temp_struct[0] = aux_struct.loc[tar_aux_time, 'animal']
    # add the date of the file
    temp_struct[1] = file_timestamp.strftime('%Y_%m_%d')
    # add the experiment number
    temp_struct[2] = aux_struct.loc[tar_aux_time, 'experiment']
    # add the ini and bin paths
    temp_struct[3] = files
    temp_struct[4] = str.replace(files, '525', '610')
    # path_struct[4][-7:-4] = '610'
    temp_struct[5] = files[:-3] + 'bin'
    temp_struct[6] = temp_struct[4][:-3] + 'bin'
    # add the lvd and eye file paths
    temp_struct[7] = join(aux_struct.loc[tar_aux_time, 'folder'], aux_struct.loc[tar_aux_time, 'lvd_file'])
    temp_struct[8] = join(aux_struct.loc[tar_aux_time, 'folder'], aux_struct.loc[tar_aux_time, 'eye1_file'])
    temp_struct[9] = join(aux_struct.loc[tar_aux_time, 'folder'], aux_struct.loc[tar_aux_time, 'eye2_file'])

    # if there's  a stim file, add it too
    if stim_file == 1:
        temp_struct[10] = stim_files[tar_stim_time]
    else:
        temp_struct[11] = join(aux_struct.loc[tar_aux_time, 'folder'], aux_struct.loc[tar_aux_time, 'txt_file'])
    # append the row to the growing list
    path_struct.append(temp_struct)

# turn the list into a dataframe for use underneath
path_struct = pd.DataFrame(path_struct, columns=['animal', 'date', 'experimentNumber', 'ini525', 'ini610', 'bin525',
                                                 'bin610', 'lvd', 'eye1', 'eye2', 'stim', 'txt'])

# print the data frame
print_full(path_struct)
# pause execution and prompt review of the dataframe before moving files
messagebox.showwarning("Warning", "Review dataframe before moving files, otherwise stop the program")


# Create folder structure and move files

# get a list with the folders already present at the master path
data_list = listdir(master_path)
# get a list of the animals in the folders to move
animal_list = np.unique(path_struct.loc[:, 'animal'])
# get the number of animals
animal_num = len(animal_list)
# get a list of the columns in the data frame
field_list = [el for el in path_struct.columns]
# remove the animal and date ones (first 2)
field_list = field_list[3:]

# for all the animals
for animal in animal_list:
    # assemble the animal path
    animal_path = join(master_path, animal)
    # check whether the current animal has a folder already. if not, make it
    if not exists(animal_path):
        mkdir(animal_path)
    # get a list of the experiments from that animal that need to be moved
    for index, files in path_struct[path_struct.animal == animal].iterrows():
        # get the dates already present
        date_list = listdir(animal_path)
        # assemble the target date path
        date_path = join(animal_path, files.date)
        # as before, if the date folder doesn't exist, create it
        if not exists(date_path):
            mkdir(date_path)
        # assemble the experiment path
        destination = join(date_path, files.experimentNumber.lstrip('0'))
        # create a folder for the experiment data is it didn't exist before (mostly to not risk overwriting data)
        if not exists(destination):
            mkdir(destination)
        # go through the fields and move the files
        for fields in field_list:
            # if the field is empty, skip the iteration
            if not files.loc[fields]:
                continue

            # assemble the source path
            source = files.loc[fields]
            # move the files
            move(source, destination)
