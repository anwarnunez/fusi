import os
import time
from importlib import reload

import numpy as np
from matplotlib import pyplot as plt

from fusi.io import spikeglx, phy, cxlabexp

def mk_block_times_contiguous(block_times, verbose=True):
    '''Concatenate a list of block times so that they are contiguous

    Parameters
    ----------
    block_times : list of 1D arrays of different sizes (n, m, ..., z).
        List of block times where each block starts at t0.

    Returns
    -------
    contiguous_times : 1D np.ndarray, (n+m+...+z)
    '''
    contiguous_times = []
    for bdx, times in enumerate(block_times):
        if bdx > 0:
            last_times = contiguous_times[bdx-1][-2:]
            times += last_times[-1] + np.diff(last_times) - times.min()
        if verbose:
            prefix = 'first block' if bdx == 0 else contiguous_times[bdx-1].max()
            print(prefix, times.min(), times.max(), times.shape)
        contiguous_times.append(times)
    return np.hstack(contiguous_times)


def post_stimulus_times(event_times, dt=0.05, duration=1.0):
    '''Compute the peri-stimulus time histogram.

    Parameters
    ----------
    event_times (1D np.ndarray): Times of event onset in [seconds]
    dt (float-like)            : Spike bin size [seconds]
    duration (float-like)      : Duration of event [seconds]

    Returns
    -------
    times (2D np.ndarray): Time-stamps for each event
    psths (3D np.ndarray): (nevents, nbins, neurons)
        Tensor of event response time-courses for all neurons
    '''
    _start = time.time()
    atimes = []
    for idx, event_time in enumerate(event_times):
        duration = duration + dt
        nsamples = int(duration / dt)
        new_times = np.arange(nsamples)*dt + event_time
        atimes.append(new_times)
    atimes = np.asarray(atimes)
    return atimes


def change_timestamps(oldtimes, oldratehz, newratehz):
    '''Change the time stamps to a new sample rate

    Parameters
    ----------
    oldtimes (np.ndarray)    : time-stamps
    oldratehz (float-like)   : original sample rate [Hz] of time-stamps
    newratehz (float-like)   : new sample rate [Hz]

    Returns
    -------
    newtimes (1D np.ndarray)
    '''
    return (oldtimes*oldratehz)/newratehz

def compute_ratehz_diffms(effective_ratehz, base_ratehz, seconds=60*60):
    '''
    '''
    nsamples = base_ratehz*seconds
    actual_duration = nsamples/effective_ratehz
    msdiff = actual_duration - seconds
    return msdiff*1000

# backwards compatibility
compute_ratehz_diff = compute_ratehz_diffms


def digital2times(times, digital_signal):
    '''Extract onset offset times from signal

    Parameters
    ----------
    times (np.ndarray or scalar):
        * Time-stamps for the digital signal (for uneven sampling)
        * Or the [Hz] sample rate for the signal  (for even sampling)
    digital_signal (np.ndarray) : Vector of 0's and 1's

    Returns
    -------
    onsets, offsets (np.ndarray) : Onsets and offsets
    '''
    ons, offs = digital2onsets(digital_signal)
    if isinstance(times, np.ndarray):
        onset_times = np.asarray([times[on] for on in ons])
        offset_times = np.asarray([times[off] for off in offs])
    else:
        sample_rate = times
        onset_times = ons/sample_rate
        offset_times = offs/sample_rate
    return onset_times, offset_times


def analog2digital(signal, threshold=2.5):
    '''Digitize an analog signal.

    Assumes a standard 5V analog signal.

    Parameters
    ----------
    signal (np.ndarray)   : continous analog signal
    threshold (float-like): points below threshold are set to zero
                            points above threhold are set to 1

    Returns
    -------
    digitized_signal (np.ndarray, np.int8):
        Data converted to zeros and ones.
    '''
    return (signal >= threshold).astype(np.int8)


def digital2onsets(digital_signal):
    '''Find the indexes of onsets and offsets of a digital signal.

    Onset is the first `on` sample.
    Offset is the first `off` sample.

    Parameters
    ----------
    digital_signal (1D np.ndarray) :
        Vector of 0's and 1's

    Returns
    -------
    onsets, offsets (1D np.ndarray):
        Two vectors, one of onsets and another with corresponding offsets

    Example
    -------
    >>> # Signal starts at 3 (1st acquired sample) ends at 7 (1st non-acq sample)
    >>> digital_signal = np.asarray([0,0,0,1,1,1,1,0,0,0])
    >>> ons, offs = digital2onsets(digital_signal)
    >>> print(ons, offs)
    [3] [7]
    >>> d2 = np.asarray([0,1,0,1,1,0,0,1,1,1,0,0,0])
    >>> ons, offs = digital2onsets(d2)
    >>> print(ons)
    [1 3 7]
    >>> print(offs)
    [ 2  5 10]
    '''
    onsets = np.where(np.diff(digital_signal) > 0)[0] + 1 # correct for np.diff
    offsets = np.where(np.diff(digital_signal) < 0)[0] + 1
    return onsets, offsets

def get_timestamps(sample_ratehz, array, offset_secs=0.0):
    '''
    '''
    if isinstance(array, np.ndarray):
        nsamples = array.shape[0]
    elif np.isscalar(array):
        nsamples = array
    else:
        msg = 'Cannot handle type: %s'%type(array)
        raise ValueError(msg)
    return np.arange(nsamples, dtype=np.uint64)/sample_ratehz + offset_secs


def timestamps2samples(sample_ratehz, timestamps, dtype=np.uint64):
    '''
    '''
    samples = timestamps*sample_ratehz
    # round first to avoid float->int even/odd mapping
    samples_uint = np.round(samples, 0).astype(dtype)
    assert np.allclose(samples, samples_uint)
    return samples_uint


def fix_onoffs_mismatches(digital_source, digital_dest, tol_nevents=1):
    '''
    '''
    tol_nsamples = np.inf

    # extract on/off indexes
    src_onsets, src_offsets = digital2onsets(digital_source)
    dst_onsets, dst_offsets = digital2onsets(digital_dest)

    nmax_onsets = min(len(src_onsets), len(dst_onsets))
    nmax_offsets = min(len(src_offsets), len(dst_offsets))

    # error in on/off digital signals must be less than X flips
    assert abs(len(src_onsets) - len(dst_onsets)) <= tol_nevents
    assert abs(len(src_offsets) - len(dst_offsets)) <= tol_nevents
    assert (abs(len(src_onsets) - len(dst_onsets)) +
            abs(len(src_offsets) - len(dst_offsets)) <= tol_nevents)

    # signals might not match because of padding
    if (len(src_onsets) != len(dst_onsets)) or (len(src_offsets) != len(dst_offsets)):
        print('src # on/off:', len(src_onsets), len(src_offsets))
        print('dst # on/off:', len(dst_onsets), len(dst_offsets))
        if (src_onsets[0] < src_offsets[0]) == (dst_onsets[0] < dst_offsets[0]):
            print('mismatch detected at end')
            ## both start on the same state, trim end
            src_onsets = src_onsets[:nmax_onsets]
            dst_onsets = dst_onsets[:nmax_onsets]
            src_offsets = src_offsets[:nmax_offsets]
            dst_offsets = dst_offsets[:nmax_offsets]
        elif (src_onsets[-1] < src_offsets[-1]) == (dst_onsets[-1] < dst_offsets[-1]):
            ## both end on the same state
            print('mismatch detected at start')
            # onsets
            src_onsets = src_onsets[::-1][:nmax_onsets][::-1]
            dst_onsets = dst_onsets[::-1][:nmax_onsets][::-1]
            src_offsets = src_offsets[::-1][:nmax_offsets][::-1]
            dst_offsets = dst_offsets[::-1][:nmax_offsets][::-1]

            # # moar verbose version
            # if len(src_onsets) < len(dst_onsets):
            #     dst_onsets = dst_onsets[1:]
            # elif len(src_onsets) > len(dst_onsets):
            #     src_onsets = src_onsets[1:]
            # # offsets
            # if len(src_offsets) < len(dst_offsets):
            #     dst_offsets = dst_offsets[1:]
            # elif len(src_offsets) > len(dst_offsets):
            #     src_offsets = src_offsets[1:]
        else:
            raise ValueError('Cannot fix mismatched on/off signals')

    # onsets and offsets must match
    assert len(src_onsets) == len(dst_onsets)
    assert len(src_offsets) == len(dst_offsets)
    return (src_onsets, src_offsets), (dst_onsets, dst_offsets)


def test_fix_mismatches():
    '''
    # mismatch at the end, so drop last flip
    A: 0100
    B: 0101

    A: 1011
    B: 1010

    # mismatch at the start
    A: 0010
    B: 1010

    A: 1101
    B: 0101
    '''
    (aon, aoff), (bon, boff) = fix_onoffs_mismatches(np.asarray([0,1,0,0]), # A
                                                     np.asarray([0,1,0,1])) # B
    assert np.allclose(aon, bon) and np.allclose(aoff, boff)
    assert (aon[0] == bon[0] == 1) and (aoff[-1] == boff[-1] == 2)

    (aon, aoff), (bon, boff) = fix_onoffs_mismatches(np.asarray([1,0,1,1]), # A
                                                     np.asarray([1,0,1,0])) # B
    assert np.allclose(aon, bon) and np.allclose(aoff, boff)
    assert (aon[0] == bon[0] == 2) and (aoff[-1] == boff[-1] == 1)


    (aon, aoff), (bon, boff) = fix_onoffs_mismatches(np.asarray([0,0,1,0]), # A
                                                     np.asarray([1,0,1,0])) # B
    assert np.allclose(aon, bon) and np.allclose(aoff, boff)
    assert (aon[0] == bon[0] == 2) and (aoff[-1] == boff[-1] == 3)

    (aon, aoff), (bon, boff) = fix_onoffs_mismatches(np.asarray([1,1,0,1]), # A
                                                     np.asarray([0,1,0,1])) # B
    assert np.allclose(aon, bon) and np.allclose(aoff, boff)
    assert (aon[0] == bon[0] == 3) and (aoff[-1] == boff[-1] == 2)

    # fail when it is very un-mached
    myfunc = lambda : fix_onoffs_mismatches(np.asarray([1,1,1,0,1]), # A
                                            np.asarray([1,0,1,0,1]), # B
                                            )

    import unittest
    tst = unittest.TestCase()
    tst.assertRaises(AssertionError, myfunc)

    # (aon, aoff), (bon, boff) =  sync.fix_onoffs_mismatches(np.asarray([1,1,1,1,0,1]), # A
    #                                                        np.asarray([0,1,0,1,0,1]), # B
    #                                                        tol_nevents=2)



def temporal_align_source2dest(times_source, digital_source,
                               times_dest, digital_dest):
    '''Align the `source` digital signal to the `destination` digital signal.

    Achieved by finding the first onset for each `source` and `dest`.
    The timestamp corresponding to that time point is set to be the same
    on both signals. The `dest` signal is kept the same and the `source` times
    are changed to match.

    Parameters
    ----------
    times_source (1D np.ndarray)   : source time stamps
    digital_source (1D np.ndarray) : vector of on(=1) and off(=0) periods

    times_dest (1D np.ndarray)     : destination time stamps
    digital_dest (1D np.ndarray)   : vector of on(=1) and off(=0) periods

    Returns
    -------
    new_source_times (1D np.ndarray):
        `source` times matching destination times.

    Examples
    --------
    >>> signal = np.asarray([0,1,0,1,1,0,0,1,1,1,0,0,0])
    >>> source_digital = np.repeat(signal, 5)
    >>> source_times = np.linspace(0, 10, len(source_digital)) + 100
    >>> dest_digital = np.repeat(signal, 20)
    >>> dest_times = np.linspace(0, 10, len(dest_digital)) + 10
    >>> newtimes = temporal_align_source2dest(source_times, source_digital, dest_times, dest_digital)
    >>> # plt.figure(); plt.plot(dest_times, dest_digital); plt.plot(newtimes, source_digital)
    '''
    # extract on/off indexes
    src_onsets, src_offsets = digital2onsets(digital_source)
    dst_onsets, dst_offsets = digital2onsets(digital_dest)

    if (len(src_onsets) != len(dst_onsets)) or (len(src_offsets) != len(dst_offsets)):
        # align the mismatched on/off cycles
        (src_onsets, src_offsets), (dst_onsets, dst_offsets) = fix_onoffs_mismatches(
            digital_source, digital_dest)

    # onsets and offsets must match
    assert len(src_onsets) == len(dst_onsets)
    assert len(src_offsets) == len(dst_offsets)

    # center source clock to first acquired event
    src_on = src_onsets[0]                       # first acquired sample index
    src_on_time = times_source[src_on]           # first acquired sample time
    source_centered = times_source - src_on_time # t=0 is first acquired sample (source clock)

    # center destination clock to first acquired event
    dst_on = dst_onsets[0]                       # onset index of dest signal
    dst_on_time = times_dest[dst_on]             # onset time of dest signal

    # t=0 is shifted to align to destination clock
    source_aligned_times = source_centered + dst_on_time
    return source_aligned_times


def get_block_from_continuous(times, digital_blocks, signals,
                              blocknum=0, padding_samples=10000,
                              onset_is_timezero=True, verbose=True):
    '''Extract a block of data from a continuous acquisition.

    Defines t=0 as the start of the block. Assumes _blocks` vector
    is 1 when acquisition is on and 0 when off.



    Parameters
    ----------
    times (1D np.ndarray)          : (n,) time-stamps
    digital_blocks (1D np.ndarray) : (n,) vector of on(=1) and off(=0) periods
    signals (np.ndarray)           : (n,p) array of data to split into blocks
    blocknum (int)                 : block index to extract
    padding_samples (scalar): pad block with this many samples before and after.

    Returns
    -------
    time_block (1D np.ndarray) : (m,) time-stamps with block onset at t=0
    signals_block (np.ndarray) : (m,p) array of block data

    Examples
    --------
    >>> # Continuous acquisition with 3 blocks of different duration
    >>> block_markers = np.asarray([0,1,0,1,1,0,0,1,1,1,0,0,0])
    >>> times = np.linspace(0, len(block_markers) - 1 , len(block_markers))
    >>> signals = np.random.randn(block_markers.shape[0],5)
    >>> block_time, block_signals = get_block_from_continuous(times, block_markers, signals, blocknum=1)
    Extracting block #1 (nblocks=3): 3.00000000-5.00000000[secs]
    >>> print(block_time)
    [-3. -2. -1.  0.  1.  2.  3.  4.  5.  6.  7.  8.]
    '''
    assert len(times) == len(digital_blocks)
    onsets, offsets = digital2onsets(digital_blocks)
    raw_on = onsets[blocknum]
    raw_off = offsets[blocknum]
    time_zero = times[raw_on]

    # one second before and after
    padding = int(padding_samples)
    on = raw_on - padding
    off = raw_off + padding

    # check for limits
    on = max(on, 0)
    off = min(off, len(times)-1)

    signals_block = signals[on:off]
    time_block = times[on:off] - time_zero if onset_is_timezero else times[on:off]

    info = (blocknum, (times[raw_off]-times[raw_on])/60., len(onsets), times[raw_on], times[raw_off])
    if verbose:
        print('Extracting: block #%i (dur=%9.05f[mins]; nblocks=%i). (%0.04f,%0.04f)[secs].'%info)
    return time_block, signals_block


def compute_effective_sampling_rate(times_source, digital_source,
                                    times_dest, digital_dest,
                                    source_sampling_ratehz=None,
                                    verbose=True):
    '''Estimate the effective sampling rate of one signal relative to the other.
    Achieved by computing the linear drift between two digital signals.

    The sampling rate of the `source` signal is modified in order to remove
    the linear drift estimated relative to `destination` signal.

    Parameters
    ----------
    times_source (1D np.ndarray)   : (n,) vector of time-stamps for source
    digital_source (1D np.ndarray) : (n,) on(=1) off(=0) digital source signal

    times_dest (1D np.ndarray)     : (m,) vector of time-stamps for target signal
    digital_dest (1D np.ndarray)   : (m,) digital destination signal

    source_sampling_ratehz (scalar or None) :
        Supposed sampling of the source signal.
        If `None`, it's computed empirically as the mean difference in time-stamps.

    Returns
    -------
    source_sampling_rate_hz (float):
        Effective sampling rate of source signal in hertz.
        This effectively removes the linear drift.

    Examples
    --------
    >>> signal = np.repeat(np.asarray([0,1,0,1,1,0,0,1,1,1,0,0,0]), 10)
    >>> sampling_rate_hz = 10.0
    >>> times_dest = np.arange(len(signal))*(1./sampling_rate_hz)
    >>> linear_drift = (sampling_rate_hz + 1e-03)
    >>> times_source = np.arange(len(signal))*(1./linear_drift)
    >>> print(times_source[:5]) # signal is fast
    [0.      0.09999 0.19998 0.29997 0.39996]
    >>> print(times_dest[:5])
    [0.  0.1 0.2 0.3 0.4]
    >>> new_sampling_hz = compute_effective_sampling_rate(times_source, signal, times_dest, signal, verbose=False)
    >>> assert np.allclose(new_sampling_hz, sampling_rate_hz) # recovers ground-truth
    >>> new_times_source = np.arange(len(times_source))*(1./new_sampling_hz)
    >>> print(new_times_source[:5])
    [0.  0.1 0.2 0.3 0.4]
    '''
    onsets_source, offsets_source = digital2onsets(digital_source)
    onsets_dest, offsets_dest = digital2onsets(digital_dest)

    if (len(onsets_source) != len(onsets_dest)) or (len(offsets_source) != len(offsets_dest)):
        # align the mismatched on/off cycles
        (onsets_source, offsets_source), (onsets_dest, offsets_dest) = fix_onoffs_mismatches(
            digital_source, digital_dest)

    assert len(onsets_source) == len(onsets_dest)
    assert len(offsets_source) == len(offsets_dest)

    onsets_difference = times_source[onsets_source] - times_dest[onsets_dest]
    offsets_difference = times_source[offsets_source] - times_dest[offsets_dest]

    Bon = np.linalg.lstsq(times_source[onsets_source][...,None],
                          onsets_difference[...,None],
                          rcond=1e-12)[0]

    Boff = np.linalg.lstsq(times_source[offsets_source][...,None],
                           offsets_difference[...,None],
                           rcond=1e-12)[0]
    if verbose:
        info = (len(onsets_source), len(offsets_source), Bon, Boff)
        print('drift: factors (nsamples=(%i,%i)): on=%0.10f, off=%0.10f'%info)

    # Use all samples
    factor = np.linalg.lstsq(np.r_[times_source[onsets_source],
                                   times_source[offsets_source]][...,None],
                             np.r_[onsets_difference,
                                   offsets_difference][..., None],
                             rcond=1e-12)[0]

    if source_sampling_ratehz is None:
        original_sampling = 1./np.mean(np.diff(times_source))
    else:
        original_sampling = source_sampling_ratehz

    newtimes = times_source*(1 - factor)
    sampling_rate = 1./np.mean(np.diff(newtimes))
    if verbose:
        info = (sampling_rate, original_sampling)
        print('Sampling: effective=%14.8f[Hz] (originally %14.8f[Hz])'%info)
        msdiff = compute_ratehz_diff(sampling_rate, original_sampling, seconds=60*30)
        print('Difference: %10.10f[ms] after 30[mins].'%msdiff)
    return sampling_rate


def remove_single_datapoint_onsets(digital_signal):
    '''Remove "on" periods consising of one single data point.

    On periods must be `1`.
    Off periods must be `0`.

    Parameters
    ----------
    digital_signal (1D np.ndarray):
        (n,) vector of digital signals

    Returns
    -------
    clean_signal (1D np.ndarray) :
        (n,) clean vector

    Examples
    --------
    >>> digital_signal = np.asarray([0,0,1,0,1,1,1,0])
    >>> remove_single_spikes(digital_signal)
    array([0, 0, 0, 0, 1, 1, 1, 0])
    >>> digital_signal = np.asarray([0,0,0,1,1,1,0])
    >>> # Do nothing if no bad datapoints found
    >>> remove_single_spikes(digital_signal) is digital_signal
    '''
    ons, offs = digital2onsets(digital_signal)
    signal_before_onset = digital_signal[ons - 1]
    signal_after_onset = digital_signal[ons + 1]
    is_bad_onsets = np.logical_and(signal_before_onset == 0,
                                signal_after_onset == 0)

    if is_bad_onsets.sum() > 0:
        clean_signal = digital_signal.copy()
        bad_datapoints = ons[is_bad_onsets]
        clean_signal[bad_datapoints] = 0
        print('Removed %i single data point onsets:'%is_bad_onsets.sum(),
              bad_datapoints)
    else:
        clean_signal = digital_signal
    return clean_signal
