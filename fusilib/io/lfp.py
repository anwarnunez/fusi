from collections import defaultdict as ddict

import numpy as np
from scipy import signal

def spectrogram(data,
                sampling_ratehz=2500,
                window_size_sec=1.0,
                step_size_sec=0.25,
                verbose=True):
    '''Compute the spectrogram in [V**2] units (power spectrum).

    Parameters
    ----------
    data : 1D np.ndarray
    sampling_ratehz : scalar
    window_size_sec : scalar
        Spectrogram window
    step_size_sec : scalar
        Spectrogram temporal step size

    Returns
    -------
    fq : 1D np.ndarray (f,)
        Frequencies
    time : 1D np.ndarray (t,)
        Time stamps
    power : 2D np.ndarray (f, t)
        Power spectrum
    '''
    window_nsamples = int(sampling_ratehz*window_size_sec)
    overlap_nsamples = window_nsamples - int(sampling_ratehz*step_size_sec)

    if verbose:
        print('Window: %s[sec] (n=%i)'%(window_size_sec,window_nsamples))
        print('Step: %s[sec] (overlap n=%i)'%(step_size_sec, overlap_nsamples))
    fq, time, power = signal.spectrogram(data,
                                         sampling_ratehz,
                                         nperseg=window_nsamples,
                                         noverlap=overlap_nsamples,
                                         scaling='spectrum',
                                         mode='magnitude',
                                         detrend='linear')
    if verbose:
        print('Time:',time[:10])
        print('deltaTime:', np.diff(time)[:10])
        print('deltaFreq:', np.diff(fq)[:10])

    return fq, time, power


def extract_bands(f, power):
    '''Compute average power across bands of interest:

    alpha+theta :   4-12Hz
    beta        :  12-30Hz
    gamma       :  30-90Hz
    high-gamma  : 110-170Hz

    Parameters
    ----------
    f : frequencies
    power : power

    Returns
    -------
    bands : dict
       Power at those bands

    Notes
    -----
    After Lima, et al., 2014
    '''
    alpha = np.logical_and(f >= 4, f < 12)
    beta = np.logical_and(f >= 12, f < 30)
    gamma = np.logical_and(f >= 30, f < 90)
    hgamma = np.logical_and(f >= 110, f < 170)

    return {'alpha' : power[alpha].mean(0),
            'beta'  : power[beta].mean(0),
            'gamma' : power[gamma].mean(0),
            'hgamma': power[hgamma].mean(0)}


def generate_channel_lfp_bands(flname,
                               sampling_ratehz=2500,
                               window_size_sec=1.0,
                               step_size_sec=0.25,
                               channels='all'):
    '''Estimate alpha, beta, gamma and high-gamma power from LFP.

    Parameters
    ----------
    flname : str
        Path to the LFP binary file
    sampling_ratehz : scalar
    window_size_sec : scalar
        Spectrogram window
    step_size_sec : scalar
        Spectrogram temporal step size

    Returns
    -------
    time : 1D np.ndarray (n,)
        Sample times
    bands : 2D np.ndarray (n, 4)
        Array containing spectral power for each band
        [alpha, beta, gamma, hgamma]
    '''
    from fusilib.io import spikeglx as glx

    mmap = glx.bin2memmap(flname)
    if channels == 'all':
        channels = range(385)

    for channel in channels:
        signal = glx.millivolt_scale_data(mmap, flname, channel)

        freq, time, power = spectrogram(signal,
                                        sampling_ratehz=sampling_ratehz,
                                        window_size_sec=window_size_sec,
                                        step_size_sec=step_size_sec)
        bands = extract_bands(freq, power)
        bands = np.asarray([bands['alpha'],
                            bands['beta'],
                            bands['gamma'],
                            bands['hgamma']])
        yield time, bands



def generate_neural_lfp_bands(flname, channels='all'):
    '''Estimate alpha, beta, gamma and high-gamma power from LFP.

    Uses methodology from Lima et al 2014, which uses different
    windows and step sizes for different bands.

    Parameters
    ----------
    flname : str
        LFP binary file

    Returns
    -------
    low_times : 1D np.ndarray, (n,)
        Time stamps for alpha and beta bands
    high_times : 1D np.ndarray, (m,)
        Time stamps for gamma and high-gamma

    bands : dict
        Dictionary containg the bands
        * alpha : (n,)
        * beta : (n,)
        * gamma : (m,)
        * hgamma : (m,)
    '''
    from fusilib.io import spikeglx as glx

    # Bands for all channels
    channel_bands = ddict(list)

    mmap = glx.bin2memmap(flname)
    if channels == 'all':
        channels = range(385)

    for chdx, channel in enumerate(channels):
        print('Working on channel: %i (%i/%i)'%(channel, chdx+1, len(channels)))

        signal = glx.millivolt_scale_lfp_data(mmap, flname, channel)

        # Match methodology from Lima , et al., 2014

        # 4-30Hz are processed at 1sec 250ms steps
        freq, low_time, power = spectrogram(signal,
                                            sampling_ratehz=2500,
                                            window_size_sec=1.0,
                                            step_size_sec=0.25,
                                            verbose=chdx==0)
        if True:
            # LFP recordings in fUSI rig have artifact
            bad = np.logical_and(freq >= 148, freq < 156)
            power[bad] = 0

        bands = extract_bands(freq, power)
        alpha, beta = bands['alpha'], bands['beta']

        # 30-170Hz are processed at 250sec 63ms steps (use 62ms instead b/c evenly divisible)
        freq, high_time, power = spectrogram(signal,
                                             sampling_ratehz=2500,
                                             window_size_sec=0.25,
                                             step_size_sec=0.062,
                                             verbose=chdx==0)
        if True:
            # LFP recordings in fUSI rig have artifact
            bad = np.logical_and(freq >= 148, freq < 156)
            power[bad] = 0

        bands = extract_bands(freq, power)
        gamma, hgamma = bands['gamma'], bands['hgamma']

        channel_bands['alpha'].append(alpha)
        channel_bands['beta'].append(beta)
        channel_bands['gamma'].append(gamma)
        channel_bands['hgamma'].append(hgamma)

    channel_bands = {k : np.asarray(v) for k, v in channel_bands.items()}
    return low_time, high_time, channel_bands
