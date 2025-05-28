import braingeneers.utils.smart_open_braingeneers as smart_open
import h5py
import pandas as pd
import numpy as np
import h5py
import zipfile
fs = 20000

def load_raw_maxwell(dataset_path: str, channel_list: list, rec_period: list, fs=20000.0):
    """
    To read the raw recording data from a maxwell hdf5 file, both new and old version
    :param dataset_path: local or s3 datapath
    :param channels: [0, len(channels)-1]. The index to maxwell recording channels.
    :param rec_period: [start, end] in seconds
    :param fs: sampling rate
    :return: ndarray of shape num_channels x (rec_period x fs), dtype = float32
    """
    # sort channels because loading from h5py needs a increasing order of indexing 
    if len(channel_list)  > 1:
        org_channels = channel_list.copy()
        channels = np.sort(org_channels)
        order = np.argsort(org_channels)
    else:
        channels = channel_list.copy()
    with smart_open.open(dataset_path, 'rb') as f:
        with h5py.File(f, 'r') as dataset:
            if 'mapping' in dataset.keys():
                signal = dataset['sig']
                gain_uV = dataset['settings']['lsb'][0] * 1e6
                mw_channels = np.array(dataset['mapping']['channel'])
                matched_chan = mw_channels[channels]  # channels from kilosort are 0-based
                # print(signal.shape)
            else:
                signal = dataset['recordings']['rec0000']['well000']['groups']['routed']['raw']
                gain_uV = dataset['recordings']['rec0000']['well000']['settings']['lsb'][0] * 1e6
                # mw_channels = np.array(dataset['wells']['well000']['rec0000']['settings']['mapping']['channel'])
                matched_chan = channels.copy()
            block_start_frame = int(fs * rec_period[0])
            block_end_frame = int(fs * rec_period[1])
            curr_signal = signal[matched_chan, block_start_frame: block_end_frame]
            # sort raw signal back to or_channels order
            if len(channel_list) > 1:
                _, raw_signal = zip(*sorted(zip(order, curr_signal)))
            else:
                raw_signal = curr_signal
            raw_signal = np.array(raw_signal).astype('float32') * gain_uV
            # print(raw_signal.shape)
    
    return raw_signal

def load_info_maxwell(dataset_path, fs=20000.0):
    with smart_open.open(dataset_path, 'rb') as f:
        with h5py.File(f, 'r') as dataset:
            if 'version' and 'mxw_version' in dataset.keys():
                version = dataset['mxw_version'][0]
                start_time = dataset['recordings']['rec0000']['well000']['start_time'][0]
                stop_time = dataset['recordings']['rec0000']['well000']['stop_time'][0]
                df = pd.DataFrame({'start_end': [start_time, stop_time]})
                time_stamp = pd.to_datetime(df['start_end'], unit='ms')
                config_df = pd.DataFrame({'pos_x': np.array(dataset['recordings']['rec0000']['well000']['settings']['mapping']['x']), 
                          'pos_y': np.array(dataset['recordings']['rec0000']['well000']['settings']['mapping']['y']),
                          'channel': np.array(dataset['recordings']['rec0000']['well000']['settings']['mapping']['channel'])})                 
                raw_frame = np.array(dataset['recordings']['rec0000']['well000']['spikes']['frameno'])
                raster_df = pd.DataFrame({'channel': np.array(dataset['recordings']['rec0000']['well000']['spikes']['channel']),
                                         'frameno': (raw_frame - raw_frame[0]) / fs})
            else:
                version = dataset['version'][0]
                time_stamp = dataset['time'][0].decode('utf-8')
                config_df = pd.DataFrame({'pos_x': np.array(dataset['mapping']['x']), 
                                    'pos_y': np.array(dataset['mapping']['y']),
                                    'channel': np.array(dataset['mapping']['channel']),
                                    'electrode': np.array(dataset['mapping']['electrode'])})
                raw_frame = np.array(dataset['proc0']['spikeTimes']['frameno'])
                rec_startframe = dataset['sig'][-1, 0] << 16 | dataset['sig'][-2, 0]
                raster_df = pd.DataFrame({'channel': np.array(dataset['proc0']['spikeTimes']['channel']),
                                      'frameno': (raw_frame - rec_startframe) / fs,
                                       'amplitude': np.array(dataset['proc0']['spikeTimes']['amplitude'])})        
    return version, time_stamp, config_df, raster_df

def load_curation(qm_path):
    with zipfile.ZipFile(qm_path, 'r') as f_zip:
        qm = f_zip.open("qm.npz")
        data = np.load(qm, allow_pickle=True)
        spike_times = data["train"].item()
        fs = data["fs"]
        train = [times / fs for _, times in spike_times.items()]
        if "config" in data:
            config = data["config"].item()
        else:
            config = None
        neuron_data = data["neuron_data"].item()
    return train, neuron_data, config, fs