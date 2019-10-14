import numpy as np
from paths import master_path
from functions_misc import multifolder, get_iframe_times, rolling_window, interp_trace, sub2ind, interp_trace_multid
from functions_load import load_lvd, load_eye_monitor_data
from functions_2pPlot import plot_2pdataset
from os.path import basename, join, dirname
from os import listdir
from scipy.io import loadmat
from scipy.signal import kaiserord, filtfilt, firwin
import pandas as pd


# select the target folders
# TODO: allow multiple folder selection across folders
folders_path = multifolder(master_path)

# define the frame averaging used
# TODO: get frame averaging automatically
frame_averaging = 1
# TODO: add progress bar

# define the type of experiment, stage or not
# TODO: get the experiment type automatically
experiment_type = 0

# for all the folders
for folders in folders_path:
    print('Loading the calcium data')
    # get the number of individual experiments combined in the folder (i.e. right and left eye)
    exp_string = basename(folders).split('_')
    exp_num = len(exp_string)

    # get the Ca file path
    ca_file = join(folders, 'suite2p', 'plane0')
    # load the calcium data
    F = np.load(join(ca_file, 'F.npy'))
    Fneu = np.load(join(ca_file, 'Fneu.npy'))
    F_chan2 = np.load(join(ca_file, 'F_chan2.npy'))
    Fneu_chan2 = np.load(join(ca_file, 'Fneu_chan2.npy'))
    # assemble a list with the data
    ca_cell_all = [[F, Fneu], [F_chan2, Fneu_chan2]]
    # load the individual ROI stats
    stat = np.load(join(ca_file, 'stat.npy'), allow_pickle=True)
    # load the options
    ops = np.load(join(ca_file, 'ops.npy'), allow_pickle=True)
    ops = ops.item()
    # load the deconvolved spikes
    sp = np.load(join(ca_file, 'spks.npy'))
    # get the iscell vector
    iscell = np.load(join(ca_file, 'iscell.npy'))
    # get the neuropil coefficient used
    neurop_coeff = ops['neucoeff']
    # get the number of image frames
    if ops.get('nframes_per_folder') is None:
        im_frames_all = np.array([ops['nframes']])
    else:
        im_frames_all = ops['nframes_per_folder']
    # initialize a frame counter
    frame_counter = 0
    # for all the experiments in this folder
    for index, experiments in enumerate(range(exp_num)):
        print('Processing file: ' + str(index))
        print('Load lvd and eye files')
        # load the im_frames for this experiment
        im_frames = im_frames_all[index]

        # load the calcium data for this experiment
        ca_cell = [[el2[:, frame_counter:frame_counter+im_frames] for el2 in el] for el in ca_cell_all]

        # assemble the root path for this experiment
        single_path = join(dirname(folders), exp_string[experiments])

        # get the number of galvo scans based on the frame averaging and the number of image frames
        num_scans_im = frame_averaging*im_frames

        # load the lvd file
        # get the path to the lvd file
        lvd_file = [join(single_path, el) for el in listdir(single_path) if el.endswith('.lvd')][0]
        # load the file contents
        lvd_data, scan_rate, num_channels, timestamp, input_range = load_lvd(lvd_file)

        # load the eye camera files
        # get the paths to the eye files
        eye1_file = [join(single_path, el) for el in listdir(single_path) if el.endswith('.eye1')][0]
        eye2_file = [join(single_path, el) for el in listdir(single_path) if el.endswith('.eye2')][0]
        # load them
        _, eye1_data = load_eye_monitor_data(eye1_file)
        # select the correct installment of the function depending on the experiment type
        if experiment_type == 0:
            _, eye2_data = load_eye_monitor_data(eye2_file, file_type='old_eye2')
        else:
            _, eye2_data = load_eye_monitor_data(eye2_file, file_type='new_eye2')

        # get the frame times
        iframe1_times, _ = get_iframe_times(eye1_data, lvd_data[:, 3])
        iframe2_times, _ = get_iframe_times(eye2_data, lvd_data[:, 3])
        # load the stage file if a stage protocol
        # TODO: implement if needed
        if experiment_type == 1:
            # get the file path
            prot_file = [join(single_path, el) for el in listdir(single_path) if el.endswith('.txt')][0]

        print('Extract imaging frame times from the lvd data')
        # trim the trace
        # scale the galvo trace to +/- 1
        lvd_data[:, 2] = lvd_data[:, 2]/np.max(lvd_data[:, 2])
        # take the derivative of the galvo trace
        diff_vector = np.diff(lvd_data[:, 2])
        # grab the vector of sign transitions in the derivative, and pad a 0 at
        # the beginning and equate to 1 to make an index vector for the original
        # trace
        peak_vector = np.hstack((0, (diff_vector[:-1] > 0) & (diff_vector[1:] < 0), 0)) == 1
        trough_vector = np.hstack((0, (diff_vector[:-1] < 0) & (diff_vector[1:] > 0), 0)) == 1
        # find the positions in time of the peaks
        peak_pos = np.argwhere(peak_vector)
        trough_pos = np.argwhere(trough_vector)
        # get the values of the peaks from the raw vector
        peak_values = lvd_data[peak_vector, 2]
        trough_values = lvd_data[trough_vector, 2]
        # filter the peak values to only the ones right at the actual peak and
        # above a defined amount of separation from the neighbors
        valid_peaks = peak_values > 0.75
        # get the valid peak positions
        valid_peakpos = peak_pos[valid_peaks]
        # for the valid troughs, use the trough immediately preceding every
        # valid peak
        valid_troughs = np.array([np.argmax(el < trough_pos) for el in valid_peakpos]) - 1

        # get the positions of the valid peaks in time. The size of the vector
        # should now match the number of frames in the imaging traces * the frame
        # averaging.
        peak_vector = peak_pos[valid_peaks]
        trough_vector = trough_pos[valid_troughs]

        # assemble a vector to look at the separation between frames
        sep_vector = np.hstack((peak_vector[:-1], trough_vector[1:]))
        sep_vector = np.hstack((np.diff(sep_vector).flatten() > 1, 0)) == 1
        # extend the frames with more than 1 separation to make it gapless
        # [since the relevant component is the mode of when the stimulus
        # happened, not the exact position of the galvo]
        peak_vector[sep_vector] = peak_vector[sep_vector] + 1

        # trim this vector from the end to match the number of imaging frames
        frame_vector = np.hstack((trough_vector, peak_vector))
        frame_vector = frame_vector[:, -num_scans_im+1:]

        # based on the position of the stimuli, determine whether the file
        # might've been cut at the beginning or the end
        # find the delta between the trigger and the last stim start. if
        # it's less than 100 frames, flag as possible early termination
        trigger_off = np.argwhere(np.diff(lvd_data[:, 3]) < -0.5)[-1]
        last_stim = np.argwhere(np.diff(lvd_data[:, 0]) < -2)[-1]
        # also find out if the first stimulus starts before the first frame
        first_stim = np.argwhere(np.diff(lvd_data[:, 0]) > 2)[0]
        if (trigger_off-last_stim) > 100:
            cut_flag = 2
        elif first_stim < frame_vector[1, 1]:
            cut_flag = 1
        else:
            cut_flag = 0

        # now trim the lvd data from the beginning of the first frame to the end
        # of the last frame
        lvd_data = lvd_data[frame_vector[0, 0]:frame_vector[-1, 1]+1, :]

        # get the frame intervals [gotta calculate from peak to peak]
    #      frame_intervals = diff[frame_vector,1,1]
        frame_intervals = np.hstack((np.diff(frame_vector[:, 0]),  frame_vector[-1, 1]+1-frame_vector[-1, 0]))
        # and the median frame time in s
        med_frame_time = np.median(frame_intervals)/scan_rate*frame_averaging
        # finally, get the median sample rate
        sample_rate = 1/med_frame_time
        # process the stim file if it is a vis protocol
        if experiment_type == 0:
            print('Processing the stimulus file')
            # get the path to the stim file
            stim_file = [join(single_path, el) for el in listdir(single_path) if el.endswith('.mat')][0]
            # load the file contents
            stim_data = loadmat(stim_file)
            # get a vector that indicates the particular stimulus protocol, stimulus
            # rep, and post stim interval for each trial [in the order in which they were
            # presented]

            # allocate memory for this vector
            trial_info = np.zeros((stim_data['Param']['stimSeq'][0][0].shape[1], 4))
            # load the vector of stimulus protocols used
            trial_info[:, 0] = np.uint16(stim_data['Param']['stimSeq'][0][0].flatten()*10)
            # get a list with the fields for the protocols
            fields_ran = list(stim_data['Param']['StimProtocols'][0][0][0].dtype.fields.keys())
            # # identify the number of stimulus protocols used
            # stimprot_num = len(fields_ran)
            # allocate memory to store the ids of the different protocols
            stimprot_ids = []
            # get the field names inside stim_data
            field_names = list(stim_data.keys())
            # exclude Param and the hidden ones
            stimprot_ids.append([el for el in field_names if el not in ['__header__', '__version__', '__globals__',
                                                                        'Param']])
            # recalculate the number of protocols
            stimprot_num = len(stimprot_ids[0])

            stimprot_ids.append([np.uint16(stim_data[el]['stim_id'][0][0][0][0]*10) for el in stimprot_ids[0]])

            # turn stimprot_ids into an array
            stimprot_ids = np.array(stimprot_ids)
            # get a vector with the sorted protocol numbers
            sorted_protocols = np.sort(stimprot_ids[1, :]).astype(np.uint16)
            
            # for all of the protocols
            for protocols in sorted_protocols:
                # get the field name of the corresponding protocol
                tar_name = stimprot_ids[0, np.uint16(stimprot_ids[1, :]) == protocols][0]
                
                if tar_name == 'DG':
                    print('DG')
                    # fill in the particular stimulus [might have to use a switch, since
                    # the names of the fields can vary depending on the stimulus. For
                    # now I'll leave it as just seq_directions]

                    # older versions use seqdirections, newer seqangles, to try both
                    try:
                        trial_info[trial_info[:, 0] == protocols, 1] = stim_data[tar_name]['seqdirections'][0][0]\
                            .flatten()
                    except ValueError:
                        trial_info[trial_info[:, 0] == protocols, 1] = stim_data[tar_name]['seqangles'][0][0].flatten()

                    # load the rep for each trial
                    rep_num = stim_data[tar_name]['n_reps'][0][0][0][0]
                    stim_num = stim_data[tar_name]['directions'][0][0][0][0]

                    rep_idx = np.array([np.arange(rep_num) for el in range(stim_num)]).T
                    trial_info[trial_info[:, 0] == protocols, 2] = rep_idx.flatten()
                    # load the post-stim interval
                    trial_info[trial_info[:, 0] == protocols, 3] = stim_data[tar_name]['poststim_time'][0][0][0][0]
                elif tar_name == 'RFM':
                    print('RFM')
                    # get the stimulus numbers in a unique sequence,
                    # combining all the conditions
                    unique_seq = stim_data[tar_name]['stimseq'][0][0]
                    trial_info[trial_info[:, 0] == protocols, 1] = unique_seq.flatten()
                    # load the rep for each trial
                    rep_num = stim_data[tar_name]['n_reps'][0][0][0][0]
                    stim_num = unique_seq.shape[1]
                    rep_idx = np.array([np.arange(rep_num) for el in range(stim_num)]).T
                    trial_info[trial_info[:, 0] == protocols, 2] = rep_idx.flatten()
                    # load the post-stim interval
                    trial_info[trial_info[:, 0] == protocols, 3] = stim_data[tar_name]['interpatch_time'][0][0][0][0]
                elif tar_name == 'LO':
                    print('LO')
                    # get the stimulus numbers in a unique sequen&ce,
                    # combining all the conditions
                    tmp = stim_data[tar_name]['paramorder'][0][0]
                    unique_seq = sub2ind(tmp)
                    trial_info[trial_info[:, 0] == protocols, 1] = unique_seq.flatten()
                    # load the rep for each trial
                    rep_num = stim_data[tar_name]['n_reps'][0][0][0][0]
                    stim_num = np.unique(tmp[0, :, :], axis=0).shape[0]
                    # stim_num = np.unique(tmp[0, :, 0]).shape[0]*np.unique(tmp[0, :, 1]).shape[0]
                    rep_idx = np.array([np.arange(rep_num) for el in range(stim_num)]).T
                    trial_info[trial_info[:, 0] == protocols, 2] = rep_idx.flatten()
                    # load the post-stim interval
                    trial_info[trial_info[:, 0] == protocols, 3] = stim_data[tar_name]['poststim_time'][0][0][0][0]
                else:
                    # trial_info = []
                    print('something else')

        # # turn trial info into integers
        # trial_info = trial_info.astype(np.uint16)
        # get the total number of trials
        trial_num = trial_info.shape[0]

        print('Process the Ca traces down to de-trended ROIs')
        # design the filter to low pass the data [based on the sampling rate
        # nyquist sampling rate
        nyq_rate = sample_rate/2.0
        # width of the filter window
        width = 0.2/nyq_rate
        # attenuation
        ripple_db = 65
        # get the order and beta coeff with a kaiser window
        N, beta = kaiserord(ripple_db, width)
        # define the cutoff frequency
        cutoff_hz = 0.8
        # get the actual filter
        taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))

        # allocate memory for the filtered, neuropil subtracted data
        f_cell = []
        # for both channels
        for chan in range(2):
            # filter the data
            filt_data = filtfilt(taps, 1.0, ca_cell[chan][0], axis=1)
            filt_npdata = filtfilt(taps, 1.0, ca_cell[chan][1], axis=1)

            # Subtract neuropil
            f_cell.append(filt_data-neurop_coeff*filt_npdata+neurop_coeff*np.median(filt_npdata))

        # calculate R[t]
        R_t = f_cell[0]/f_cell[1]

        # detrend
        # define the length in time of the window [in s]
        window_time = 14

        # perform the detrending
        R_detrend = R_t - rolling_window(R_t, np.round(sample_rate*window_time).astype(int), np.percentile, 8,
                                         axis=1)
        print('Align each frame with the lvd info')
        # get a trace of lvd_data binned [via mode] to match the imaging frames
        # generate the label vector to bin the frame times according to the
        # frame averaging
        label_vec = pd.DataFrame(data=
                                 np.vstack(
                                     (np.repeat(np.arange(frame_intervals.shape[0]/frame_averaging).astype(np.uint16),
                                        frame_averaging),
                                      frame_intervals)).T,
                                 columns=['label', 'times'])
        # calculate the edges of each frame in time
        edge_vector = np.hstack((0, np.cumsum(np.array(label_vec.groupby('label').sum()))))

        # generate the index vector for each frame
        idx_vec = np.digitize(np.arange(lvd_data.shape[0]), edge_vector)
        # TODO: find a file with this messed up to try the section
        # check for gaps in the idx vector. if any, interpolate
        gap_list = np.argwhere(np.diff(idx_vec) > 1)+1
        
        if gap_list.shape[0] > 0:
            while gap_list.shape[0] > 0:
                nanc = gap_list[0]
                # add an extra value in the index vector and the data vector
                # [via interpolation]
                idx_vec = np.hstack((idx_vec[:nanc], idx_vec[nanc-1]+1, idx_vec[nanc:]))
                # generate the interpolated data point
                interp_point = interp_trace(np.arange(1, 3), lvd_data[nanc-1:nanc+1, :], 1.5)

                lvd_data = np.hstack((lvd_data[:nanc, :], interp_point, lvd_data[nanc:, :]))
                # refind the gaps
                gap_list = np.argwhere(np.diff(idx_vec) > 1)+1

        # if visual stimulation, get the periods with stimulation from the
        # lvd file
        if experiment_type == 0:
            # generate the data frame
            dac_time = pd.DataFrame(data=np.vstack((idx_vec, lvd_data[:, 0].astype(np.uint16))).T,
                                    columns=['label', 'stim'])
            # use the index vector to take the mode of the samples for each frame
            stim_perframe = np.array(dac_time.groupby('label').agg(lambda x: pd.Series.mode(x)[0]))
            # stim_perframe = splitapply[@mode,lvd_data[1,:],idx_vec]

        print('Align each frame with the eye cam info')
        # label the eye data frames with the corresponding microscope frame
        idx_eye1vec = np.digitize(iframe1_times, edge_vector)
        idx_eye2vec = np.digitize(iframe2_times, edge_vector)

        # depending on the type of eye file, select the relevant columns
        if eye1_data.shape[0] < 10:
            # select the columns with the X and Y of the eye plus the pupil area
            eye1_cols = eye1_data[6:9, :]
            # also define the corresponding type of file
            cam1_type = 'eye'
        else:
            # select the columns with the object position, size, and stage
            # status
            eye1_cols = eye1_data[2:, :]
            # also define the corresponding type of file
            cam1_type = 'stage'
        # do the same for the second camera
        if eye2_data.shape[0] < 10:
            eye2_cols = eye2_data[6:9, :]
            cam2_type = 'eye'
        else:
            eye2_cols = eye2_data[2:, :]
            cam2_type = 'stage'
        # check whether the last frame is missing. If so, copy the second to
        # last frame
        if np.max(idx_eye1vec) < im_frames:
            max_idx = np.argmax(idx_eye1vec)
            idx_eye1vec = np.hstack((idx_eye1vec[:max_idx], im_frames, idx_eye1vec[max_idx+1]))
            eye1_cols = np.hstack((eye1_cols[:, :max_idx],
                                   np.expand_dims(eye1_cols[:, max_idx], axis=1),
                                   np.expand_dims(eye1_cols[:, max_idx+1], axis=1)
                                   ))
        # # NaN the values from the points after the last frame
        # elif np.max(idx_eye1vec) > im_frames:
        #     idx_eye1vec[idx_eye1vec > im_frames] = np.nan
        if np.max(idx_eye2vec) < im_frames:
            max_idx = np.argmax(idx_eye2vec)
            idx_eye2vec = np.hstack((idx_eye2vec[:max_idx], im_frames, idx_eye2vec[max_idx+1]))
            eye2_cols = np.hstack((eye2_cols[:, :max_idx],
                                   np.expand_dims(eye2_cols[:, max_idx], axis=1),
                                   np.expand_dims(eye2_cols[:, max_idx+1], axis=1)
                                   ))
        # # NaN the values from the points after the last frame
        # elif np.max(idx_eye2vec) > im_frames:
        #     idx_eye2vec[idx_eye2vec > im_frames] = np.nan

        # continue with the eye data
        # fill in NaNs at the beginning of the vector and the data to be able to
        # use the splitapply function
        # get the first frame present
        first1 = idx_eye1vec[0]
        first2 = idx_eye2vec[0]

        # remake the vector
        nan_idx_eye1vec = np.hstack((np.arange(1, first1), idx_eye1vec))
        nan_idx_eye2vec = np.hstack((np.arange(1, first2), idx_eye2vec))
        # also NaN any values that skip frames
        delta_cam1frame = np.diff(nan_idx_eye1vec) > 1
        delta_cam2frame = np.diff(nan_idx_eye2vec) > 1

        # extend the data vector by the same amount with NaNs
        nan_eye1_cols = np.hstack((np.zeros((eye1_cols.shape[0], first1-1))*np.nan, eye1_cols))
        nan_eye2_cols = np.hstack((np.zeros((eye2_cols.shape[0], first2-1))*np.nan, eye2_cols))

        # find out if there are isolated positions [i.e. not at the
        # beginning or the end] that have NaNs
        inner_nan1 = np.argwhere(delta_cam1frame) + 1

        # consider the entire trace if doing a stage experiment
        if experiment_type == 0:
            inner_nan2 = np.argwhere(delta_cam2frame[99:-100]) + 100
        else:
            inner_nan2 = np.argwhere(delta_cam2frame) + 1

        # if there are inner nans, insert interpolated frames in the data.
        # If there are more than 10 [and it's not the stage file, cause
        # lower frame rate], throw and error though
        assert not (inner_nan1.shape[0] > 10) or (inner_nan2.shape[0] > 10 and experiment_type == 0), \
            'Too many skipped frames in the eye cameras'
        # find where the continuous NaNs begin at the end
        last_nan1 = np.argwhere(np.diff(np.isnan(nan_idx_eye1vec)))
        if last_nan1.size > 0:
            last_nan1 = last_nan1[-1]
        else:
            last_nan1 = -1
        # TODO: find file with this messed up and test the segment
        # mostly fixed, needs to be validated on another data set or two - MM 14.10.19
        if inner_nan1.size > 0:
            while inner_nan1.size > 0:
                nanc = inner_nan1[0][0]
                # add an extra value in the index vector and the data vector
                # [via interpolation]
                nan_idx_eye1vec = np.hstack((nan_idx_eye1vec[:nanc],
                                            nan_idx_eye1vec[nanc-1]+1, nan_idx_eye1vec[nanc:]))
                # generate the interpolated data point
                interp_point = interp_trace_multid(np.arange(1, 3), nan_eye1_cols[:, nanc-1:nanc+1], 1.5)
                nan_eye1_cols = np.hstack((nan_eye1_cols[:, :nanc],
                                           np.expand_dims(interp_point, axis=1),
                                           nan_eye1_cols[:, nanc:]))

                delta_cam1frame = np.diff(nan_idx_eye1vec) > 1

                inner_nan1 = np.argwhere(delta_cam1frame[:last_nan1])+1
                last_nan1 = last_nan1 + 1

        # find where the continuous NaNs begin at the end
        last_nan2 = np.argwhere(np.diff(np.isnan(nan_idx_eye2vec)))
        if last_nan2.size > 0:
            last_nan2 = last_nan2[-1]
        else:
            last_nan2 = -1

        if inner_nan2.size > 0:
            while inner_nan2.size > 0:
                nanc = inner_nan2[0][0]
                # add an extra value in the index vector and the data vector
                # [via interpolation]
                nan_idx_eye2vec = np.hstack((nan_idx_eye2vec[:nanc],
                                            nan_idx_eye2vec[nanc-1]+1, nan_idx_eye2vec[nanc:]))
                # generate the interpolated data point
                interp_point = interp_trace_multid(np.arange(1, 3), nan_eye2_cols[:, nanc-1:nanc+1], 1.5)
                nan_eye2_cols = np.hstack((nan_eye2_cols[:, :nanc],
                                           np.expand_dims(interp_point, axis=1),
                                           nan_eye2_cols[:, nanc:]))

                delta_cam2frame = np.diff(nan_idx_eye2vec) > 1
                # consider the entire trace if doing a stage experiment
                if experiment_type == 0:
                    inner_nan2 = np.argwhere(delta_cam2frame[99:-100])+100
                else:
                    inner_nan2 = np.argwhere(delta_cam2frame[:last_nan2])+1
                    last_nan2 = last_nan2 + 1

        # Now NaN the extrema if they contain deltas larger than 1
        delta_cam1frame = np.diff(nan_idx_eye1vec) > 1
        delta_cam2frame = np.diff(nan_idx_eye2vec) > 1
        nan_idx_eye1vec = nan_idx_eye1vec.astype(float)
        nan_idx_eye2vec = nan_idx_eye2vec.astype(float)
        nan_idx_eye1vec[np.hstack((0, delta_cam1frame)) == 1] = np.nan
        nan_idx_eye2vec[np.hstack((0, delta_cam2frame)) == 1] = np.nan
        # trim the traces to contain the same number of scope frames
        min_frames = np.min([np.nanmax(nan_idx_eye1vec), np.nanmax(nan_idx_eye2vec)])
        nan_idx_eye1vec[nan_idx_eye1vec > min_frames] = np.nan
        nan_idx_eye2vec[nan_idx_eye2vec > min_frames] = np.nan

        # assemble a data frame to then apply groupby
        cam1_time = pd.DataFrame(data=nan_eye1_cols.T, columns=['eye_x', 'eye_y', 'pupil_diam'])
        cam1_time['label'] = nan_idx_eye1vec.astype(np.uint32)
        eye1_perframe = np.array(cam1_time.groupby('label').mean())
        cam2_time = pd.DataFrame(data=nan_eye2_cols.T, columns=['eye_x', 'eye_y', 'pupil_diam'])
        cam2_time['label'] = nan_idx_eye2vec.astype(np.uint32)
        eye2_perframe = np.array(cam2_time.groupby('label').mean())
        # concatenate the individual cells across eyes together in a single
        # array
        cam_data = np.hstack((eye1_perframe, eye2_perframe))

        # remove frames after the last imaging frame
        if cam_data.shape[0] > im_frames:
            cam_data = cam_data[:im_frames, :]

        print('Calculate dRoR')
        # TODO: add the stage version
        # get the stim starts
        stim_start_ori = np.array([el[0] for el in np.argwhere(np.diff(stim_perframe, axis=0) > 2)]).astype(float)
        # and the stim ends
        stim_end_ori = np.array([el[0] for el in np.argwhere(np.diff(stim_perframe, axis=0) < -2)]).astype(float)

        # # if a stage experiment and spont act, add those limits to the start
        # # and end
        # if experiment_type == 1 and np.any(trial_info[:, 1] == 1):
        #     stim_start_ori = cat[2,stim_start_ori,find[frame_data[1,:]==1,1,'first']]
        #     stim_end_ori = cat[2,stim_end_ori,find[frame_data[1,:]==1,1,'last']]

        # initialize stim_start and stim_end [and do nothing if they are
        # the same length]
        stim_start = stim_start_ori
        stim_end = stim_end_ori
        # if the experiment is incomplete [i.e. trials are lost at the
        # beginning or the end]
        if stim_start_ori. shape[0] < stim_end_ori.shape[0]:
            # this means there is a starts missing at the beginning. Pad with
            # NaN
            stim_start = np.hstack((np.nan, stim_start_ori))
            # also kill the matching end
            stim_end[0] = np.nan
        elif stim_start_ori.shape[0] > stim_end_ori.shape[0]:
            # this means there is an end missing at the end. Pad with NaN
            stim_end = np.hstack((stim_end_ori, np.nan))
            # also kill the matching start
            stim_start[-1] = np.nan

        # determine whether there are trials missing
        # TODO: test the stim trigger padding
        if stim_start.shape[0] != trial_num:
            # based on the cut_flag, deal with truncated files
            if cut_flag == 1:
                # start is truncated
                # pad the beginning of the stim_start and stim_end with
                # NaNs
                stim_start = np.hstack((np.zeros((trial_num-stim_start.shape[0]))*np.nan, stim_start))
                stim_end = np.hstack((np.zeros((trial_num-stim_end.shape[0]))*np.nan, stim_end))
            elif cut_flag == 2:
                # end is truncated
                # pad the end of stim_start and stim_end with NaNs
                stim_start = np.hstack((stim_start, np.zeros((trial_num-stim_start.shape[0]))))
                stim_end = np.hstack((stim_end, np.zeros((trial_num-stim_end.shape[0]))))

        # determine the number of cells in the experiment
        cell_num = R_detrend.shape[0]

        # initialize the structure
        dRoR = {}

        if experiment_type == 0:
            # get a vector with the protocol id numbers
            prot_id = stimprot_ids[1].astype(np.uint16)

            # for all the stimulus protocols
            for protocols in np.arange(stimprot_num):
                # get the name of the protocol
                prot_name = stimprot_ids[0][prot_id == sorted_protocols[protocols]][0]
                # load the name of the protocol
                dRoR[prot_name] = {}
                # get the info for the trials in this protocol
                dRoR[prot_name]['trial_info'] = trial_info[trial_info[:, 0] == sorted_protocols[protocols], :]
                # get the number of trials in the protocol
                subtrial_num = dRoR[prot_name]['trial_info'].shape[0]
                # get the starts and ends corresponding to these trials
                prot_start = stim_start[trial_info[:, 0] == sorted_protocols[protocols]]
                prot_end = stim_end[trial_info[:, 0] == sorted_protocols[protocols]]

                # determine the number of frames to take for R0
                r0_time = dRoR[prot_name]['trial_info'][0, 3]
                r0_frames = np.ceil(sample_rate * r0_time).astype(np.uint16)

                # # get the min trial duration plus the r0 frames
                # trial_duration = (r0_frames + np.min(prot_end-prot_start)).astype(np.uint16)
                
                # define the vector of trial durations [including the r0
                # time] and accounting for the indexing difference
                trial_vec = (r0_frames + (prot_end-prot_start)).astype(np.uint16)
                if prot_name == 'LO':
                    # get the max trial duration plus the r0 frames
                    trial_duration = np.max(trial_vec).astype(np.uint16)
                elif prot_name == 'RFM':
                    # get the min trial duration plus the r0 frames at
                    # beginning and end
                    trial_duration = np.min(trial_vec) + r0_frames
                    trial_vec = (np.ones(subtrial_num)*trial_duration).astype(np.uint16)
                else:
                    # get the min trial duration plus the r0 frames
                    trial_duration = np.min(trial_vec)
                    trial_vec = (np.ones(subtrial_num)*trial_duration).astype(np.uint16)

                # load the rep for each trial
                rep_num = np.unique(dRoR[prot_name]['trial_info'][:, 2]).shape[0]
                # also the number of stimuli
                stim_angles, stim_idx = np.unique(dRoR[prot_name]['trial_info'][:, 1], return_inverse=True)
                stim_num = stim_angles.shape[0]

                # allocate memory for the trials
                trial_mat = np.zeros((cell_num, trial_duration, rep_num, stim_num))
                # allocate memory for the cam data [the last dimension is the
                # two cameras, each with x and y eye position and pupil area]
                trial_cam = np.zeros((trial_duration, rep_num, stim_num, 6))

                # for each trial
                for trials in np.arange(subtrial_num):
                    # get the current rep
                    reps = dRoR[prot_name]['trial_info'][trials, 2].astype(np.uint16)
                    # get the current stim
                    stim = stim_idx[trials]
                    # determine the trial start
                    trial_start = prot_start[trials]-r0_frames
                    # if the start is a NaN,
                    if np.isnan(trial_start):
                        # NaN also the cell in the trial_mat
                        trial_mat[:, :, reps, stim] = np.nan
                        trial_cam[:, reps, stim, :] = np.nan
                    else:
                        # turn it into integer for indexing
                        trial_start = trial_start.astype(np.uint16)
                        # transfer the dRoR info
                        # load the trial in the matrix, including the r0 period
                        trial_mat[:, :trial_vec[trials], reps, stim] = \
                            R_detrend[:, np.arange(trial_start, trial_start + trial_vec[trials])]
                        # get the corresponding cam info
                        trial_cam[:trial_vec[trials], reps, stim, :] = \
                            cam_data[np.arange(trial_start, trial_start + trial_vec[trials]), :]

                # calculate dRoR and store the main structure
                R0 = np.reshape(np.median(trial_mat[:, :r0_frames, :, :], axis=1), (cell_num, 1, rep_num, stim_num))
                dRoR[prot_name]['data'] = (trial_mat - R0)/R0
                # save the camera data
                dRoR[prot_name]['camData'] = trial_cam
                # save the ball data
                dRoR[prot_name]['ballData'] = lvd_data[:, 1]
        print('Save data')
        # assemble the dictionary with metadata, including ROIs
        meta_data = {'caData': {}, 'frameAve': frame_averaging, 'camTypes': [cam1_type, cam2_type]}
        meta_data['caData']['ops'] = ops
        meta_data['caData']['stat'] = stat[iscell[:, 0] == 1]
        meta_data['caData']['stat'] = sp[iscell[:, 0] == 1, :]
        meta_data['caData']['filename'] = ca_file

        if experiment_type == 0:
            meta_data['stimData'] = stim_data

        # save the dRoR and stimulus sequence paired with each frame
        save_path = join(single_path, 'preProcessed.npz')
        # savefile.create_dataset('data', data=dRoR)
        # savefile.create_dataset('metadata', data=meta_data)
        np.savez(save_path, data=dRoR, metadata=meta_data)

        # plot the data and save the figure
        fig_list = plot_2pdataset(dRoR)

        for fig, name in zip(fig_list, dRoR.keys()):
            fig.savefig(join(single_path, 'traces_' + name + '.png'), bbox_inches='tight')

        # update the frame counter
        frame_counter += im_frames
print('yay')
