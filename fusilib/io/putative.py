import numpy as np
from scipy import signal, interpolate

def spike_duration(wave):
    '''Computes the total duration of the spike.

    Parameters
    ----------
    wave : 1D np.ndarray (m,)
        Spike waveform

    Notes
    -----
    Follows methodology of Bartho et al 2004.
    From the negative peak spike amplitude to the next positive peak.

    Returns
    -------
    width : scalar
        Duration of the peak.
    height : scalar
    lpt, rpt : scalars
        left is the time of spike start
        right is the time of spike end
    '''
    interpolator = interpolate.interp1d(np.arange(len(wave)), wave)
    newtimes = np.linspace(0, len(wave)-1, len(wave)*10)
    hwave = interpolator(newtimes)

    amin = hwave.argmin()
    amax = hwave[amin:].argmax() # + 1
    # semantics:
    height, lpt, rpt = hwave[amin], newtimes[amin], newtimes[amin+amax]
    width = rpt - lpt
    print(width, height, lpt, rpt)
    return np.abs(width), height, lpt, rpt


def spike_fwhm(wave, rel_height=1/2.):
    '''Computes the temporal width of the spike.

    Parameters
    ----------
    wave : 1D np.ndarray (m,)
        Spike waveform
    rel_height : scalar
        Percentage of peak at which to compute width
        0.5 = FWHM

    Notes
    -----
    Follows methodology of Bartho et al 2004.
    Computed the full-width at half max

    Returns
    -------
    width : scalar
        Full-width at half max by default.
    height : scalar
    lpt, rpt : scalars
        left is the first time the FWHM is reached
        right is the time the FWHM is reached on the way down
    '''
    interpolator = interpolate.interp1d(np.arange(len(wave)), wave)
    newtimes = np.linspace(0, len(wave)-1, len(wave)*10)
    hwave = interpolator(newtimes)

    height = (hwave.min() - wave[0])*rel_height
    height += wave[0]
    idxs = (hwave <= height).nonzero()[0]
    lpti, rpti = min(idxs), max(idxs)
    lpt, rpt = newtimes[rpti], newtimes[lpti]
    width = rpt - lpt
    print(width, height, lpt, rpt)
    return np.abs(width), height, lpt, rpt

def isexcitatory(wave, duration_ms=0.45, fudge_ms=0.01):
    '''Is this the waveform of an excitatory neuron?

    Uses spike duration as criterion (default: excitatory > 0.46ms)

    Parameters
    ----------
    wave : 1D np.ndarray (m,)
        Spike waveform
    duration_ms : scalar
        A waveform with a duration longer than this
        is considered excitatory. Defaults to 0.45 ms
    fudge_ms : scalar
        Defaults to 0.01 ms

    Returns
    -------
    excitatory : bool
    '''

    fwhm = samples2ms(spike_fwhm(wave)[0])
    duration = samples2ms(spike_duration(wave)[0])
    #return fwhm > 0.2 and duration > 0.45
    return duration > (duration_ms + fudge_ms) # 0.46


def isinhibitory(wave, duration_ms=0.45, fudge_ms=0.01):
    '''Is this the waveform of an inhibitoryibitory neuron?

    Uses spike duration as criterion (default: inhibitory < 0.44ms)

    Parameters
    ----------
    wave : 1D np.ndarray (m,)
        Spike waveform
    duration_ms : scalar
        A waveform with a duration longer than this
        is considered excitatory. Defaults to 0.45 ms
    fudge_ms : scalar
        Defaults to 0.01 ms

    Returns
    -------
    inhibitory : bool
    '''
    fwhm = samples2ms(spike_fwhm(wave)[0])
    duration = samples2ms(spike_duration(wave)[0])
    #return fwhm < 0.2 and duration < 0.45
    return duration < (duration_ms + fudge_ms) # 0.44


def putative_neuron_type(wave):
    '''What neuron class is this?

    Parameters
    ----------
    wave : 1D np.ndarray (m,)
        Spike waveform

    Returns
    -------
    neuronclass : str
        * excitatory : putative exc neuron
        * inhibitory : putative inh neuron
        * none       : not clearly either
    '''
    ntype = 'none'
    if isexcitatory(wave):
        ntype = 'excitatory'
    elif isinhibitory(wave):
        ntype = 'inhibitory'
    return ntype

def samples2ms(x):
    '''Convert ephys samples to ms

    Assumes value "x" is expressed in 30,000Hz
    '''
    return x/30000*1000


def find_histogram_split(spike_duration_ms, maxima=(0.25, 0.7)):
    '''For a bimodal histogram of spike durations, finds the minimum between the two maxima.
    '''
    hh, edges = np.histogram(spike_duration_ms, bins=25, range=(0.1, 1.1))
    hh = np.where(hh < 15, 0, hh) # Threshold count
    minpeaks = signal.find_peaks(-hh)[0]
    minima = edges[minpeaks]
    print(minima)
    if maxima is None:
        maxpeaks = signal.find_peaks(hh, height=10)[0]
        maxima = edges[maxpeaks]
        print(maxima)
        assert len(maxima) == 2
    maxl, maxr = maxima
    for minval in minima:
        if np.logical_and(minval <= maxr,
                          minval >= maxl):
            return minval

    return np.nan
