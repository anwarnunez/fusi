'''Code for resampling signals

20191014: Anwar O. Nunez-Elizalde
'''
import numpy as np


def square_resample(window_size, data, oldtimes, newtimes, nlobes=3, **kwargs):
    '''Resample data using a Lanczos kernel over the specified window size

    Parameters
    ----------
    window_size (float)     : Time window in [seconds] over which to compute the filter
    data (2D np.ndarray)    : Original data of dimensions: (`oldtimes`, `samples`)
    oldtimes (1D np.ndarray): Time stamps of the data [seconds]
    newtimes (1D np.ndarray): New time points of resampled data [seconds]
    nlobes (int)            : Total number of lanczos lobes is `2*nlobes - 1`.
    **kwargs                : Passed to `lanczos_kernel` (e.g. causal=True)

    Returns
    -------
    resampled_data (2D np.ndarray) : (`newtimes`, `samples`)
    '''
    cutoff_hz = 1./(window_size/2.)
    ndata = np.zeros((len(newtimes), data.shape[1]), dtype=data.dtype)
    for sampleidx, newtime in enumerate(newtimes):
        times = newtime - oldtimes
        samples = np.abs(times) <= (window_size*2.0)
        kernel = np.ones(len(samples))
        ndata[sampleidx] = np.dot(data[samples].T, kernel)
        # Older version (~2x slower in tests)
        # kernel = lanczos_kernel(times,
        #                         cutoff_hz,
        #                         nlobes=nlobes)
        # ndata[sampleidx] = np.dot(data.T, kernel)
    return ndata


def gaussian_kernel(times, sigma=1.0):
    '''
    '''
    from tikreg import kernels
    vals = np.exp(-0.5*times**2/(2*sigma**2))
    return vals


def lanczos_kernel(times, cutoff_hz, nlobes=3,
                   causal=False, anticausal=False, normalize=True):
    '''Compute a Lanczos kernel at the specified time points and frequencies.

    Parameters
    ----------
    times (1D np.ndarray) : Time stamps in [seconds].
    cutoff_hz (scalar) : Low-pass frequency cutoff.
        This defines the location of the first lanczos lobe.
        Temporally, the first right lobe is located at `T=1/cutoff_hz` and so
        the cutoff defines the temporal lobes over which the filter is applied
        as `w = 2*(1/cutoff_hz)`.
    nlobes (int) : Specify the lobes to one side.
        Total number of lanczos lobes is `2*nlobes - 1`.
    causal (bool)     : Only values *after* the zero-point are kept.
    anticausal (bool) : Only values *before*  the zero-point are kept.

    Returns
    -------
    lanczos (1D np.ndarray) : Specified lanczos filter

    Notes
    -----
    https://en.wikipedia.org/wiki/Lanczos_resampling
    '''
    times = times * cutoff_hz
    val = ((nlobes * np.sin(np.pi*times) * np.sin(np.pi*times/nlobes)) /
           (np.pi**2 * times**2))
    if causal:
        val[times < 0] = 0.0
    if anticausal:
        val[times > 0] = 0.0

    val[times == 0] = 1.0
    val[np.abs(times) > nlobes] = 0.0
    norm = len(np.abs(times) <= nlobes) if normalize else 1.0
    return val / norm


def lanczos_resample(window_size, data, oldtimes, newtimes, nlobes=3, **kwargs):
    '''Resample data using a Lanczos kernel over the specified window size

    Parameters
    ----------
    window_size (float)     : Time window in [seconds] over which to compute the filter
    data (2D np.ndarray)    : Original data of dimensions: (`oldtimes`, `samples`)
    oldtimes (1D np.ndarray): Time stamps of the data [seconds]
    newtimes (1D np.ndarray): New time points of resampled data [seconds]
    nlobes (int)            : Total number of lanczos lobes is `2*nlobes - 1`.
    **kwargs                : Passed to `lanczos_kernel` (e.g. causal=True)

    Returns
    -------
    resampled_data (2D np.ndarray) : (`newtimes`, `samples`)
    '''
    cutoff_hz = 1./(window_size/2.)
    ndata = np.zeros((len(newtimes), data.shape[1]), dtype=data.dtype)
    for sampleidx, newtime in enumerate(newtimes):
        times = newtime - oldtimes
        samples = np.abs(times) <= (window_size*2.0)
        kernel = lanczos_kernel(times[samples],
                                cutoff_hz,
                                nlobes=nlobes,
                                **kwargs)

        ndata[sampleidx] = np.dot(data[samples].T, kernel)
        # Older version (~2x slower in tests)
        # kernel = lanczos_kernel(times,
        #                         cutoff_hz,
        #                         nlobes=nlobes)
        # ndata[sampleidx] = np.dot(data.T, kernel)
    return ndata


def lanczos_even_resample(data, oldtimes, newdt=0.05, **kwargs):
    '''
    Parameters
    ----------
    data (2D np.ndarray)     : (`nsamples` `nunits`)
    oldtimes (1D np.ndarray) : (`nsamples`,)
        Time stamps of the data [seconds]
    newdt (float)            : Temporal resolution of new data [seconds]

    **kwargs (dict, optional): Passed to `lanczos_resample`.

    '''
    start = oldtimes.min()
    end = oldtimes.max()
    duration = end - start
    newsamples = int(np.ceil(duration/newdt))

    window_size = newdt              # [seconds]
    newtimes = np.arange(newsamples)*newdt + start
    newdata = lanczos_resample(window_size, data, oldtimes, newtimes, **kwargs)
    return newtimes, newdata


def lanczos_even_resample_duration(data, oldtimes, newdt=0.05, event_duration=1.0, **kwargs):
    '''
    Parameters
    ----------
    data (2D np.ndarray)     : (`nsamples` `nunits`)
    oldtimes (1D np.ndarray) : (`nsamples`,)
        Time stamps of the data [seconds]
    newdt (float)            : Temporal resolution of new data [seconds]
    event_duration           : duration of one `oldtime/data` event [seconds]

    **kwargs (dict, optional): Passed to `lanczos_resample`.

    '''
    start = oldtimes.min()
    end = oldtimes.max()
    duration = end - start
    newsamples = int(np.ceil(duration/newdt))

    window_size = newdt              # [seconds]
    newtimes = np.arange(newsamples)*newdt + start

    olddt = newdt/3.0           # 3x faster sampling than new DT
    upsample = int(event_duration/newdt)

    updata = np.repeat(data, upsample, axis=0)
    uptimes = np.arange(upsample)*olddt
    upoldtimes = np.hstack([nt + uptimes for nt in oldtimes])
    newdata = lanczos_resample(window_size,
                               updata,
                               upoldtimes, newtimes, **kwargs)
    return newtimes, newdata


def lanczos_resample_atdelays(data, oldtimes, newtimes,
                              window_size=None, delays=[0], kwargs={}):
    '''Return a resampled window of responses at different delays around the newtimes.

    Parameters
    ----------
    data (2D np.ndarray)     : (`nsamples` `nunits`)
    oldtimes (1D np.ndarray) : (`nsamples`)
        Time stamps of the data [seconds]
    newtimes (1D np.ndarray) : (`nsamples`)
        Time stamps of the data [seconds]
    delays (1D np.ndarray)   :
        Time points relative to new times [seconds]
    window_size (float, optional):
        Defaults to smallest difference between delays
    kwargs (dict, optional): Passed to `lanczos_resample`.

    Returns
    -------
    newdata (3D np.ndarray) : (`ndelays`, `newtimes`, `nunits`)
    '''
    newdata = np.zeros((len(delays), len(newtimes), data.shape[1]),
                       dtype=data.dtype)

    # minimum
    if window_size is None:
        window_size = np.diff(delays).min()

    for delayidx, delay in enumerate(delays):
        print('Working on %i/%i delay=%0.04f[sec] window=%0.04f[sec]' % (delayidx+1,
                                                                         len(delays),
                                                                         delay,
                                                                         window_size))
        newdata[delayidx] = lanczos_resample(window_size,
                                             data,
                                             oldtimes,
                                             newtimes + delay,
                                             **kwargs)

    return newdata


def cutoff2window(cutoff_hz):
    '''window_size is in seconds
    '''
    window = (1./cutoff_hz)*2
    return window


def window2cutoff(window_size):
    '''window_size is in seconds
    '''
    cutoff_hz = 1./(window_size/2.)
    return cutoff_hz
