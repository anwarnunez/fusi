'''Functions to load binary data stored with SpikeGLX.

Anwar O. Nunez-Elizalde (Dec 2019).
'''
import os
from pathlib import Path
from collections import defaultdict as ddict

import numpy as np


def load_data_chunk(flname, start_time=0, sample_duration=-1, voltage=True):
    '''Load binary files saved by SpikeGLX

    The binary file may contain data from a neuropixel probe (Imec) or DAQ.

    Parameters
    ----------
    flname (str)                : Location of binary file
    start_time (float-like)     : Start reading from this time point [seconds]
    sample_duration (float-like): Read [seconds]. Default `-1`, reads all data.
    voltage (bool)              : Convert data to voltage (default True)

    Return
    ------
    data (2D np.ndarray)

    Example
    -------
    >>> bin_path = '/DATA_ROOT/CR020/2019-11-22/ephys/2019-11-22_CR020_g0/'
    >>> npfile = os.path.join(bin_path, '2019-11-22_CR020_g0_imec0', '2019-11-22_CR020_g0_t0.imec0.ap.bin')
    >>> data_neuropixels = load_data_chunk(npfile, sample_duration=2.0) # doctest: +SKIP
    Loading 2.000000[secs] 44.06MB (385, 60000) @ 30000.00000000Hz: [0:60000]
    >>> nifile = os.path.join(bin_path, '2019-11-22_CR020_g0_t0.nidq.bin')
    >>> data_nidaq = load_data_chunk(nifile, sample_duration=2.0)       # doctest: +SKIP
    Loading 2.000000[secs] 0.11MB (3, 20000) @ 10000.00000000Hz: [0:20000]
    '''
    dtype = np.int16
    meta = read_metadata(flname)
    nchannels = meta['nSavedChans']
    sample_rate = meta['imSampRate'] if meta['typeThis'] == 'imec'\
                  else meta['niSampRate']

    # samples to read
    nsamples = -1 if sample_duration == -1 else int(sample_duration*sample_rate)
    start_sample = int(sample_rate*start_time)
    offset_bytes = (nchannels*start_sample)*np.dtype(dtype).itemsize

    # display some information
    size = (nchannels*nsamples)*np.dtype(dtype).itemsize/(2.**20) \
           if sample_duration != -1 else os.path.getsize(flname)/(2.**20)
    info = (sample_duration, size, nchannels, nsamples,
            sample_rate, start_sample, start_sample+nsamples)
    print('Loading %0.06f[secs] %0.2fMB (%i, %i) @ %0.8fHz: [%i:%i] '%info)

    # read file directly (avoid np.memmap memory leakage on *nix)
    with open(flname, 'r') as file_object:
        # skip manually for numpy<=1.16 compatibility
        file_object.seek(offset_bytes)
        data = np.fromfile(file_object, dtype=dtype, count=nchannels*nsamples)
        data = data.reshape(nsamples, nchannels).T

    if voltage:
        # Convert the raw data to millivolts
        scalings = get_data2voltage_scalings(flname)
        channel_numbers = (np.arange(data.shape[0]) if (meta['typeThis'] != 'imec')\
                           else get_imec_original_channel_numbers(flname))

        data = np.asarray(data, dtype=np.float32)
        for chidx, chnum in enumerate(channel_numbers):
            data[chidx] *= scalings[chnum] # [volts]
    return data


def millivolt_scale_lfp_data(memmap, flname, channel):
    '''Load LFP data channel and convert to millivolts.

    NB: Tested only on LFP binary files. Check for AP.

    Parameters
    ----------
    memmap : np.memmap
        From function `bin2memmap`
    flname : str
        Uses binary file name to load metadata
    channel : int

    Returns
    -------
    data : np.ndarray
    '''
    meta = read_metadata(flname)

    nchannels = memmap.shape[0]
    scalings = get_data2voltage_scalings(flname)
    channel_numbers = (np.arange(nchannels) if (meta['typeThis'] != 'imec')\
                       else get_imec_original_channel_numbers(flname))

    data = np.asarray(memmap[channel], dtype=np.float32)
    # Convert the raw data to voltage
    for chidx, chnum in enumerate(channel_numbers):
        if chidx == channel:
            data *= scalings[chnum] # [volts]
    return data*1e6



def read_metadata(flname):
    '''Load metadata corresponding to a binary file saved by SpikeGLX

    Finds and loads the meta-data file (*.meta) of a binary file (*.bin).

    Parameters
    ----------
    flname (str) : Binary file e.g. `my_data_g0_t0.imec0.ap.bin`.

    Return
    ------
    meta (dict)  : Dictionary of key-value pairs containing
                   the parsed meta-data file: `my_data_g0_t0.imec0.ap.meta`.
    '''
    assert os.path.isfile(flname)
    flroot = os.path.splitext(flname)[0]
    flmeta = '%s.meta'%flroot
    assert os.path.exists(flmeta)
    meta = dict()
    with open(flmeta, 'r') as fl:
        for line in fl.readlines():
            key, content = line.split('=')
            key = key.lstrip('~')
            value = content.strip()
            try:
                value = float(value)
                # map to int if possible
                value = int(value) if value == int(value) else value
            except:
                pass
            meta[key] = value
    return meta


def get_imec_original_channel_numbers(flname):
    '''Find the original channel number for each saved channel.

    This is needed in order to apply the correct gains to the correct
    channels as stored in the binary file.

    Parameters
    ----------
    flname (str): Binary file name (*.bin)

    Returns
    -------
    channel_numbers (1D np.ndarray):
        Vector with the original channel numbers.

    Example
    -------
    >>> bin_path = '/DATA_ROOT/CR020/2019-11-22/ephys/2019-11-22_CR020_g0/'
    >>> binfl = os.path.join(bin_path, '2019-11-22_CR020_g0_imec0', '2019-11-22_CR020_g0_t0.imec0.ap.bin')
    >>> original_channel_numbers = get_imec_original_channel_numbers(binfl)
    >>> # Note the last channel stored in the binary file (384) corresponds
    >>> # to the digital channel whose original index is 768.
    >>> print(len(original_channel_numbers))
    384
    >>> original_channel_numbers[-10:]
    array([374, 375, 376, 377, 378, 379, 380, 381, 382, 768])

    Notes
    -----
    One can specify a subset of channels to save with SpikeGLX.
    The channels as found in the binary file will correspond to this
    specified order. The metadata contains the mapping between
    the binary file order and the original channel number.
    '''
    meta = read_metadata(flname)
    assert meta['typeThis'] == 'imec'
    assert meta['snsSaveChanSubset'] != 'all'

    channel_slices = meta['snsSaveChanSubset'].split(',')
    orig_channel_numbers = []
    for cslice in channel_slices:
        index = list(map(int, cslice.split(':')))
        if len(index) == 2:
            start, end = index
            orig_channel_numbers += range(start,end)
        elif len(index) == 1:
            orig_channel_numbers.append(index[0])
        else:
            raise ValueError('Unexpected indexing:', index)
    orig_channel_numbers = np.asarray(orig_channel_numbers)
    return orig_channel_numbers


def get_data2voltage_scalings(flname):
    '''Compute the scaling factor needed to map raw data to voltage.
    Works for both Imec and NI data.

    Parameters
    ----------
    flname (str): Binary file name (*.bin)

    Returns
    -------
    data_scalings (dict):
        Dictionary mapping the scaling needed for each original channel
        {original_channel_number : scaling}
    '''
    meta = read_metadata(flname)
    int2volts = meta['imAiRangeMax']/512.0 if meta['typeThis'] == 'imec'\
                else meta['niAiRangeMax']/32768.0

    if meta['typeThis'] == 'imec':
        orig_channel_numbers = get_imec_original_channel_numbers(flname)

        # AP and LFP gains
        assert 'imDatPrb_dock' not in meta # NP 2.0 has 80 AP gain
        # load gains and remove 0=header and -1=trailing ")"
        gains_table = meta['imroTbl'].split(')')[1:-1]
        nchannels = len(gains_table)

        gains = ddict(lambda : int2volts/1.0)
        for idx, channel_gains in enumerate(gains_table):
            # remove leading "(" and convert elements to float
            channel_gains = map(float, channel_gains.lstrip('(').split())
            channel_num, _, _, ap_gain, lfp_gain, _ = channel_gains
            channel_num = int(channel_num)
            gains[channel_num] = ap_gain
            gains[channel_num + nchannels] = lfp_gain

        data_scalings = {}
        for chidx, chnum in enumerate(orig_channel_numbers):
            # convert to volts
            data_scalings[chnum] = int2volts / gains[chnum]
    else:
        ##############################
        # NI DAQ
        ##############################
        # XXX: multiplexed, multiplexed analog, simple analog, digital???
        MN, MA, XA, DW = map(int, meta['snsMnMaXaDw'].split(','))
        # no support for multiplexed channels
        assert MN == 0 and MA == 0
        data_scalings = {i : int2volts/1.0 for i in range(3)}
    return data_scalings


def get_channel_gains(flname):
    '''Load the gains applied to all channels stored in binary file.
    Works for both Imec and NI binary data.

    Parameters
    ----------
    flname (str): Binary file name (*.bin)

    Returns
    -------
    channel_gains (dict): Channel gains
        {original_channel_number : gain}
    '''
    meta = read_metadata(flname)
    int2volts = meta['imAiRangeMax']/512.0 if meta['typeThis'] == 'imec'\
                else meta['niAiRangeMax']/32768.0

    if meta['typeThis'] == 'imec':
        # AP and LFP gains for IMEC NP
        assert 'imDatPrb_dock' not in meta # NP 2.0 has 80 AP gain
        # load gains and remove 0=header and -1=trailing ")"
        gains_table = meta['imroTbl'].split(')')[1:-1]
        nchannels = len(gains_table)

        gains = ddict(lambda : 1.0)
        for idx, channel_gains in enumerate(gains_table):
            # remove leading "(" and convert elements to float
            channel_gains = map(float, channel_gains.lstrip('(').split())
            channel_num, _, _, ap_gain, lfp_gain, _ = channel_gains
            channel_num = int(channel_num)
            gains[channel_num] = ap_gain
            gains[channel_num + nchannels] = lfp_gain
    else:
        # NI board gains
        # XXX: multiplexed, multiplexed analog, simple analog, digital???
        MN, MA, XA, DW = map(int, meta['snsMnMaXaDw'].split(','))
        # no support for multiplexed channels
        assert MN == 0 and MA == 0
        gains = {i : 1.0 for i in range(3)}
    return gains


def get_spikeglx_memmap(flname):
    '''Memory-map object for binary data stored with SpikeGLX.

    Infers the number of channels in the *.bin file by reading
    the corresponding metadata file (*.meta). Assumes SpikeGLX
    defaults (i.e. F-order, int16).

    Parameters
    ----------
    flname (str): Binary file name (*.bin)

    Returns
    -------
    data_mmap (np.memmap) : 2D of shape (nchannels, nsamples)
    '''
    metadata = read_metadata(flname)
    nchannels = metadata['nSavedChans']
    mmap = bin2memmap(flname, nchannels=nchannels)
    return mmap


def bin2memmap(flname, nchannels=385, dtype=np.int16, order='F'):
    '''Wrapper to `np.memmap` with SpikeGLX defaults for Neuropixels.
    '''
    item_bytes = np.dtype(dtype).itemsize
    total_nbytes = os.path.getsize(flname)
    nsamples = int(total_nbytes/(item_bytes*nchannels))
    assert nsamples == total_nbytes/(item_bytes*nchannels) # sanity check
    mmap = np.memmap(flname, dtype=dtype, shape=(nchannels, nsamples),
                     order=order, mode='r')
    return mmap
