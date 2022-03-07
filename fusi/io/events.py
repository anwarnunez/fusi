'''
'''
import numpy as np
import fusi.io.spikes2 as iospikes

time_locked_events_matrix = iospikes.time_locked_spike_matrix

def time_locked_delay_matrix(event_ids,
                             oldtimes,
                             newtimes,
                             dt,
                             delay_window=(0, 2),
                             verbose=False):
    '''
    Parameters
    ----------
    event_ids (1D np.ndarray): Vector containing unique `nevents` (n,)
    oldtimes (1D np.ndarray) : Event onsets (n,)
    newtimes (1D np.ndarray) : New sample times (n_newtimes,)
    dt (scalar)              : New sample rate
    delay_window (tuple)     : Delays for ``newtimes``: (delay_min, delay_max)
        If ``0`` or ``None``, no delays are applied.

    Returns
    -------
    delay_times (1D np.ndarray) : Delays in seconds (ndelays,)
    delay_matrix (3D np.ndarray): Delay matrix of shape (ndelays, n_newtimes, nevents)

    Examples
    --------
    >>> dels, dmat = events_object.get_delay_matrix(np.random.rand(100),delay_window=(0,1), dt=0.1)
    >>> dmat.shape
    (10, 100, 4)
    '''
    if (delay_window == 0) or (delay_window is None):
        delay_window = (0, dt)

    delmin = int(min(delay_window)/dt)
    delmax =  int(max(delay_window)/dt)
    delays = np.arange(delmin, delmax)
    delay_times = delays*dt
    ndelays = len(delays)

    if verbose: print('Delays [sec]:',delay_times)

    delay_matrix = []
    for ddx, tdelay  in enumerate(delay_times):
        if verbose: print('Working on delay %2i/%i: %10.05f[sec]'%(ddx+1, ndelays, tdelay))

        design_matrix = time_locked_events_matrix(oldtimes,
                                                  event_ids,
                                                  newtimes - tdelay,
                                                  duration=dt,
                                                  dtype=np.float32)
        delay_matrix.append(design_matrix)
    delay_matrix = np.asarray(delay_matrix)
    return delay_times, delay_matrix



class EventsFromTrain(object):
    '''Easy computation of events responses.

    >>> events_object = EventsFromTrain(times_vector, evid_vector)
    '''
    def __init__(self, event_times,
                 event_ids,
                 dt=0.05,
                 nevents=None,
                 event_values=None):
        '''
        '''
        assert isinstance(event_times, np.ndarray)
        assert isinstance(event_ids, np.ndarray)

        # sort in time
        order = np.argsort(event_times)
        event_times = event_times[order]
        event_ids = event_ids[order]

        if event_values is not None:
            event_values = event_values[order]

        if nevents is None:
            nevents = len(np.unique(event_ids))

        # under the hood, this uses the probe spikes machinery ;)
        self.view = iospikes.ProbeSpikes(event_times,
                                         event_ids,
                                         nclusters=nevents,
                                         spike_counts=event_values)

        self.event_times = event_times
        self.event_ids = event_ids
        self.nevents = nevents
        self.nonsets = len(event_times)
        self.dt = dt

        # get sparse matrix representation of data
        self.sparse_matrix = self.view.get_sparse_spike_matrix()
        self.MBsize = self.view.sparse_matrix.data.nbytes/(2**20)

    def __call__(self, event_times, duration, dt=None):
        '''Sample the events at the requested times and durations

        Parameters
        ----------
        event_times : Vector of timestamps in seconds
        duration    : Duration after event onset
        dt          : Bin size in seconds

        Returns
        -------
        times (ntimepoints, ndelays) :
        delay_matrix (ntimepoints, ndelays, nevents) :
        '''
        dt = self.dt if dt is None else dt
        times, delays = self.view.psth(event_times, dt=dt, duration=duration)
        delays = delays.transpose(1,0,2) # (ntimepoints, ndelays, nevents)
        times = times.T                   # (ntimepoints, ndelays)
        return times, delays

    def __repr__(self):
        info = (__name__, type(self).__name__, self.nonsets, self.nevents, self.MBsize)
        return '<%s.%s (nonsets=%i, nevents=%i) [%0.02fMB]>'%info

    def get_delay_matrix(self, newtimes, delay_window=(0,2), dt=None, verbose=False):
        '''Construct a delay matrix for these events

        Horizontally stacking the delays yields a design matrix.

        Parameters
        ----------
        newtimes (1D np.ndarray):
            Sample the onsets at these times (ntimepoints,)
        delay_window (tuple):
            Delays in time [sec]: (delay_min, delay_max)
            If ``0`` or ``None``, no delays are applied.
        dt (scalar, optional): New sample rate [sec]
            Defaults to ``self.dt``

        Returns
        -------
        delay_times (1D np.ndarray) : Delays in seconds (ndelays,)
        delay_matrix (3D np.ndarray): Delay matrix (ndelays, ntimepoints, nevents)
        '''
        dt = self.dt if dt is None else dt
        if verbose: print('Sample rate: %f[sec]'%dt)
        delay_times, delay_matrix = time_locked_delay_matrix(self.event_ids,
                                                             self.event_times,
                                                             newtimes,
                                                             dt=dt,
                                                             delay_window=delay_window,
                                                             verbose=verbose)
        return delay_times, delay_matrix



class Events(EventsFromTrain):
    '''Easy computation of events responses.
    '''
    def __init__(self, event_onsets, dt=0.05, event_values=None):
        '''
        event_onsets : list
        '''
        if isinstance(event_onsets[0], (float, int)):
            # a single event as a list or array
            event_onsets = [event_onsets]

        event_times = []
        event_ids = []
        event_vals = []
        for edx, onsets in enumerate(event_onsets):
            event_times.append(np.asarray(onsets))
            event_ids.append(np.ones(len(onsets))*edx)
            if event_values is not None:
                event_vals.append(np.asarray(event_values[edx]))

        event_ids = np.hstack(event_ids)
        event_times = np.hstack(event_times)
        nevents = len(event_onsets)

        if event_values is not None:
            event_vals = np.hstack(event_vals)

        counts = None if event_values is None else event_vals
        super(type(self), self).__init__(event_times,
                                         event_ids,
                                         dt=dt,
                                         nevents=nevents,
                                         event_values=counts)
