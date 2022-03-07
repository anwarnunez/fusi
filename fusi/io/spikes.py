'''Code to generate spike trains.

Anwar O. Nunez-Elizalde (2020)
'''
import time

import numpy as np
from scipy import sparse

# from fusi import utils as futils

##############################
# Helper functions
##############################

def lowest_uint(nsamples):
    '''Lowest unsigned integer value that fits nsamples
    '''
    nbytes = np.log2(nsamples)
    dtype = np.uint64
    if nbytes < 8:
        dtype = np.uint8
    elif nbytes < 16:
        dtype = np.uint16
    elif nbytes < 32:
        dtype = np.uint32
    elif nbytes < 64:
        dtype = np.uint64
    else:
        raise OverflowError(nsamples)
    return dtype


def iterator_func(*args, **kwargs):
    '''Makes `tqdm` an optional requirement.
    '''
    return args[0]
    # try:
    #     from tqdm import tqdm
    #     return tqdm(*args, **kwargs)
    # except ImportError:
    #     return args[0]
    # raise ValueError('Unknown')


##############################
# Core functions
##############################

def mua_clusters_from_spike_matrix(spike_matrix,
                                   cluster_depths,
                                   chunk_sizeum=100,
                                   min_depthum=0,
                                   max_depthum=3840,
                                   step_sizeum=None,
                                   good_clusters=None,
                                   dtype=np.float32):
    '''Build a multi-unit activity matrix

    Parameters
    ----------
    spike_matrix  : np.ndarray (n, v)
    cluster_depths: np.ndarray (v,)
        Position of each cluster along the probe in [um]

    chunk_sizeum : int
    min_depthum  : int
    max_depthum  : int
    step_sizeum  : optional, int
    good_clusters: optional, int

    Returns
    -------
    n_mua_units : np.ndarray (m,)
        Number of clusters in MUA chunk
    mua_matrix  : np.ndarray (n, m)
        Multi-unit activity matrix counted as cluster spike sums
    '''
    step = chunk_sizeum if step_sizeum is None else step_sizeum
    nchunks = int(np.ceil((max_depthum - min_depthum)/step))
    nsamples, nclusters = spike_matrix.shape

    good_clusters = np.ones(nclusters, dtype=np.bool) \
        if good_clusters is None else good_clusters

    mua_matrix = np.zeros((nsamples, nchunks), dtype=dtype)
    n_mua_units = np.zeros(nchunks, dtype=np.int64)

    all_clusters = []

    for chunkidx in iterator_func(range(nchunks),
                                  unit='chunk',
                                  total=nchunks):
        min_depth = chunkidx*step + min_depthum
        max_depth = min(min_depth + chunk_sizeum, max_depthum + 1e-05)

        valid_clusters = np.logical_and(cluster_depths >= min_depth,
                                        cluster_depths < max_depth)

        valid_clusters = np.logical_and(valid_clusters, good_clusters)
        all_clusters.append(valid_clusters)

    return np.asarray(all_clusters)


def mua_from_spike_cluster_matrix(spike_matrix,
                                  cluster_depths,
                                  chunk_sizeum=100,
                                  min_depthum=0,
                                  max_depthum=3840,
                                  step_sizeum=None,
                                  good_clusters=None,
                                  dtype=np.float32):
    '''Build a multi-unit activity matrix

    Parameters
    ----------
    spike_matrix  : np.ndarray (n, v)
    cluster_depths: np.ndarray (v,)
        Position of each cluster along the probe in [um]

    chunk_sizeum : int
    min_depthum  : int
    max_depthum  : int
    step_sizeum  : optional, int
    good_clusters: optional, int

    Returns
    -------
    n_mua_units : np.ndarray (m,)
        Number of clusters in MUA chunk
    mua_matrix  : np.ndarray (n, m)
        Multi-unit activity matrix counted as cluster spike sums
    '''
    step = chunk_sizeum if step_sizeum is None else step_sizeum
    nchunks = int(np.ceil((max_depthum - min_depthum)/step))
    nsamples, nclusters = spike_matrix.shape

    good_clusters = np.ones(nclusters, dtype=np.bool) \
        if good_clusters is None else good_clusters

    mua_matrix = np.zeros((nsamples, nchunks), dtype=dtype)
    n_mua_units = np.zeros(nchunks, dtype=np.int64)

    # Keep an internal copy to modify
    cluster_depths = cluster_depths.copy()
    nan_depths = np.isnan(cluster_depths)
    cluster_depths[nan_depths] = -1.0
    for chunkidx in iterator_func(range(nchunks),
                                  unit='chunk',
                                  total=nchunks):
        min_depth = chunkidx*step + min_depthum
        max_depth = min(min_depth + chunk_sizeum, max_depthum + 1e-05)

        valid_clusters = np.logical_and(cluster_depths >= min_depth,
                                        cluster_depths < max_depth)

        valid_clusters = np.logical_and(valid_clusters, good_clusters)
        # Exclude nan depths
        valid_clusters[nan_depths] = False
        n_mua_units[chunkidx] = valid_clusters.sum()
        mua_matrix[:, chunkidx] = spike_matrix[:, valid_clusters].sum(-1)
    return n_mua_units, mua_matrix


def time_bins_from_timestamps(times, dt=0.01, t0=None, duration=None):
    '''Discretize the time vector in to bins of fixed duration

    Parameters
    ----------
    times : 1D np.ndarray (n,)
        Vector of time stamps
    dt    : scalar
        Desired temporal resolution (i.e. bin duration)
    t0    : optional, scalar
        Onset of the first bin
        If None, t0 is time of the first bin (i.e. a multiple of `dt`)
    duration : optional, scalar
        Length of time

    Returns
    -------
    newtimes : 1D np.ndarray (m,)
        The discritized times

    Examples
    --------
    >>> assert newtimes[0] == t0
    >>> assert newtimes[-1] == t0 + duration + dt
    '''
    if t0 is None:
        # first bin to include a spike
        t0 = (times.min() // dt)*dt
    if duration is None:
        duration = times.max() - t0

    # include the end of the bin (+dt) as a sample
    nsamples = np.ceil(duration/dt) + 1
    newtimes = np.arange(nsamples, dtype=np.float64)
    newtimes *= dt
    newtimes += t0
    return newtimes


def sparse_matrix_from_cluster_times(spike_times, spike_clusters, nclusters=None):
    '''Sparse matrix representation of all spikes and all clusters.

    Parameters
    ----------
    spike_times    : 1D float np.ndarray (n,)
        Time of spikes in [seconds]
    spike_clusters : 1D int np.ndarray (n,)
        Vector containing the cluster ID for every spike
    nclusters      : optional, int
        If None, defaults to the maximum cluster ID.
        If int, assumes this is the maximum number of clusters

    Returns
    -------
    sparse_matrix : 2D bool sparse.csr_matrix (n, nclusters)
    '''
    if nclusters is None:
        nclusters = spike_clusters.max() + 1
    shape = (len(spike_times), nclusters)
    sparse_matrix = sparse.csr_matrix((np.ones_like(spike_times),
                                       (np.arange(len(spike_times)), spike_clusters)),
                                      shape, dtype=np.bool)
    return sparse_matrix


def spike_count_matrix_from_sparse(spike_times, sparse_matrix,
                                   dt=0.001, t0=0,
                                   duration=None, dtype=None):
    '''Matrix of spike counts for all clusters

    Parameters
    ----------
    spike_times   : 1D np.ndarray (n,)
    sparse_matrix : 2D sparse matrix (n, nclusters)

    dt : float-like
        Bin size in [secs]
    t0 : float-like
        Time from which to start counting spikes in [secs]
        Defaults to 0. If `t0 is None`, then t0 is defined
        by the onset of the first bin containing a spike.
    duration : float-like
        Length of time in [secs] after `t0` to count spikes
    dtype : optional
        Defaults to using a small unsigned integer datatype

    Returns
    -------
    newtimes     : 1D np.ndarray (n,)
        Bin onset times for the spike matrix
    spike_matrix : 2D np.ndarray (n, m)
        Matrix containing spike counts for all `m` clusters.
    '''
    nclusters = sparse_matrix.shape[1]

    # compute time bins
    bin_time_edges = time_bins_from_timestamps(spike_times, dt=dt,
                                               t0=t0, duration=duration)
    # bin_indexes = np.searchsorted(spike_times, bin_time_edges, side='right')
    bin_indexes = np.digitize(bin_time_edges, spike_times, right=True)

    # minimum uint for the maximum number of spikes in a bin
    if dtype is None:
        max_nspikes = (bin_indexes[1:] - bin_indexes[:-1]).max()
        dtype = lowest_uint(max_nspikes)

    # build time and spike matrix
    newtimes = bin_time_edges[:-1] # sample times are the start of bins
    spike_matrix = np.zeros((len(newtimes), nclusters), dtype=dtype)

    for tdx, (start, end) in iterator_func(enumerate(zip(bin_indexes[:-1], bin_indexes[1:])),
                                           '%s. Generating spike count matrix'%__name__,
                                           total=len(newtimes)):
        if start == end:
            # no spikes
            continue
        spike_matrix[tdx] = sparse_matrix[start:end].sum(0)
    return newtimes, spike_matrix

def event_related_responses(spike_times, spike_clusters, event_times,
                            duration=1.0, dt=0.01,
                            nclusters=None, dtype=None):
    '''Count spikes for a period after event onset at the requested temporal resolution

    Spikes are counted for `duration` seconds after `event_time` with `dt` resolution.

    Parameters
    ----------
    spike_times    : 1D float np.ndarray (n,)
        Time of spikes in [seconds]
    spike_clusters : 1D int np.ndarray (n,)
        Vector containing the cluster ID for every spike
    event_times    : 1D float np.ndarray (m,)
        Onset of event in [seconds]
    duration : scalar or 1D np.ndarray (m,)
        Duration for all events or for each event in [seconds]
        Defaults to 1.0 [seconds]
    dt : float-like
        Bin size in [secs]

    nclusters : optional, int (`k`)
        If None, defaults to the maximum cluster ID.
        If int, assumes this is the maximum number of clusters
    dtype : optional
        Defaults to using a small unsigned integer datatype

    Returns
    -------
    spike_matrix : 3D np.ndarray (m, t, k)
        Matrix containing spike counts for all `k` clusters,
        for all `m` events for `t` samples after event onset.
    '''
    # setup the data
    sparse_matrix = sparse_matrix_from_cluster_times(spike_times,
                                                     spike_clusters,
                                                     nclusters=nclusters)
    nclusters = sparse_matrix.shape[1]
    event_nsamples = int(np.ceil(duration/dt))
    spike_matrix = np.zeros((len(event_times), event_nsamples, nclusters), dtype=dtype)
    event_time_bins = np.zeros((len(event_times), event_nsamples), dtype=np.float)

    # determine minimum dtype
    # bin_edge_left  = np.searchsorted(spike_times, event_times, side='right')
    bin_edge_left  = np.digitize(event_times, spike_times, right=True)
    # bin_edge_right = np.searchsorted(spike_times, event_times + duration, side='right')
    bin_edge_right = np.digitize(event_times + duration, spike_times, right=True)
    if dtype is None:
        max_nspikes = (bin_edge_right - bin_edge_left).max()
        dtype = lowest_uint(max_nspikes)

    # event durations
    if not isinstance(duration, (np.ndarray, list)):
        duration = np.repeat(duration, len(event_times))
    assert len(event_times) == len(duration)

    # go through the spikes
    for edx, event_time in iterator_func(enumerate(event_times),
                                         '%s. event time-locked spike responses'%__name__,
                                         total=len(event_times)):
        # compute the time bin edgeso
        time_bins = time_bins_from_timestamps(np.atleast_1d(event_time),
                                              dt=dt,
                                              t0=event_time,
                                              duration=duration[edx])
        # bin_time_edges  = np.searchsorted(spike_times, time_bins, side='right')
        bin_time_edges  = np.digitize(time_bins, spike_times, right=True)
        event_time_bins[edx] = time_bins[:-1]

        for tdx, (start, end) in enumerate(zip(bin_time_edges[:-1], bin_time_edges[1:])):
            if start == end:
                # no spikes
                continue
            spike_matrix[edx, tdx] = sparse_matrix[start:end].sum(0)
    return event_time_bins, spike_matrix


def time_locked_spike_matrix(spike_times, spike_clusters,
                             event_times, duration=1.0,
                             nclusters=None, dtype=None, ):
    '''Spike matrix sampled at the event times

    Parameters
    ----------
    spike_times    : 1D float np.ndarray (n,)
        Time of spikes in [seconds]
    spike_clusters : 1D int np.ndarray (n,)
        Vector containing the cluster ID for every spike
    event_times    : 1D float np.ndarray (m,)
        Onset of event in [seconds]
    duration : scalar or 1D np.ndarray (m,)
        Bin size in [seconds].
        Duration for all events or for each event in [seconds]

    nclusters : optional, int (`k`)
        If None, defaults to the maximum cluster ID.
        If int, assumes this is the maximum number of clusters
    dtype : optional
        Defaults to using a small unsigned integer datatype


    Returns
    -------
    spike_matrix : 2D np.ndarray (m, k)
        Matrix containing spike counts for all `k` clusters.
    '''
    sparse_matrix = sparse_matrix_from_cluster_times(spike_times,
                                                     spike_clusters,
                                                     nclusters=nclusters)

    nclusters = sparse_matrix.shape[1]

    # bin_edge_left  = np.searchsorted(spike_times, event_times, side='right')
    bin_edge_left  = np.digitize(event_times, spike_times, right=True)
    # bin_edge_right = np.searchsorted(spike_times, event_times + duration, side='right')
    bin_edge_right = np.digitize(event_times + duration, spike_times, right=True)

    if dtype is None:
        max_nspikes = (bin_edge_right - bin_edge_left).max()
        dtype = lowest_uint(max_nspikes)
    spike_matrix = np.zeros((len(event_times), nclusters), dtype=dtype)

    for tdx, (start, end) in iterator_func(enumerate(zip(bin_edge_left, bin_edge_right)),
                                           '%s. time-locked spike matrix'%__name__,
                                           total=len(event_times)):
        if start == end:
            # no spikes
            continue
        spike_matrix[tdx] = sparse_matrix[start:end].sum(0)
    return spike_matrix


def spike_count_matrix(spike_times, spike_clusters, nclusters=None,
                       dt=0.001, t0=0, duration=None, dtype=None):
    '''Generate a spike count matrix from the raw spike clusters.

    Parameters
    ----------
    spike_times    : 1D float np.ndarray (n,)
        Time of spikes in [seconds]
    spike_clusters : 1D int np.ndarray (n,)
        Vector containing the cluster ID for every spike
    nclusters      : optional, int
        If None, defaults to the maximum cluster ID.
        If int, assumes this is the maximum number of clusters
    dt : float-like
        Bin size in [secs]
    t0 : float-like
        Time from which to start counting spikes in [secs]
        Defaults to 0. If `t0 is None`, then t0 is defined
        by the onset of the first bin containing a spike.
    duration : float-like
        Length of time in [secs] after `t0` to count spikes
    dtype    : optional
        Defaults to using a small unsigned integer datatype

    Returns
    -------
    newtimes     : 1D np.ndarray (n,)
        Bin onset times for the spike matrix
    spike_matrix : 2D np.ndarray (n, m)
        Matrix containing spike counts for all `m` clusters.
    '''
    sparse_matrix = sparse_matrix_from_cluster_times(spike_times,
                                                     spike_clusters,
                                                     nclusters=nclusters)

    newtimes, spike_matrix = spike_count_matrix_from_sparse(
        spike_times, sparse_matrix,
        dt=dt, t0=t0, duration=duration, dtype=dtype)

    return newtimes, spike_matrix

##############################
# core classes
##############################

class ProbeSpikes(object):
    '''Easy computation of spike responses.
    '''
    def __init__(self, spike_times, spike_clusters, nclusters=None, good_clusters=None, cluster_depths=None):
        '''
        spike_times
        spike_clusters
        '''
        self.times = spike_times
        self.clusters = spike_clusters
        if nclusters is None:
            nclusters = int(spike_clusters.max() + 1)
        self.nclusters = nclusters
        self.nspikes = len(spike_times)
        self.good_clusters = good_clusters
        self.cluster_depths = cluster_depths

        # get sparse matrix representation of data
        self.sparse_matrix = self.get_sparse_spike_matrix()
        self.MBsize = self.sparse_matrix.data.nbytes/(2**20)

    def __repr__(self):
        info = (__name__, type(self).__name__, self.nspikes, self.nclusters, self.MBsize)
        return '<%s.%s (nspikes=%i, nclusters=%i) [%0.02fMB]>'%info

    def get_sparse_spike_matrix(self):
        '''Sparse matrix representation of all spikes and all clusters.

        Returns
        -------
        sparse_matrix : 2D bool sparse.csr_matrix (nspikes, nclusters)
        '''
        sparse_matrix = sparse_matrix_from_cluster_times(self.times,
                                                         self.clusters,
                                                         self.nclusters)
        return sparse_matrix

    def bin_spike_matrix(self, dt=0.05, start_time=None):
        '''Compute a spike matrix with fixed width time bins.

        Parameters
        ----------
        dt (float-like)       : Duration of time bins in [seconds]
        start_time (float-like: Onset of first bin.
            If None, defaults to time of first spike

        Returns
        -------
        new_times (1D np.ndarray)       : bin onsets
        bin_spike_matrix (2D np.ndarray): spike counts
        '''
        # basics
        if start_time is None:
            start_time = self.times.min()

        new_times, bin_spike_matrix = self.spike_count_matrix(dt=dt, t0=start_time)
        return new_times, bin_spike_matrix

    def event_response(self, event_onset=0.0, duration=1.0, dt=0.05):
        '''Compute spike bin counts after event.

        Parameters
        ----------
        event_onset (float-like): time of event onset in [seconds]
        duration (float-like)   : get spikes these many [seconds] after event
        dt (float-like)         : bin the spikes at this time interval [seconds]

        Returns
        -------
        new_times (1D np.ndarray) : bin times
        bin_matrix (2D np.ndarray): (nbins, nclusters)
            Matrix containing spike counts
        '''
        event_spike_times, event_spike_matrix = self.psth(np.atleast_1d(event_onset),
                                                          dt=dt, duration=duration)
        return event_spike_times[0], event_spike_matrix[0]

    def psth(self, event_times, dt=0.05, duration=1.0):
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
        event_spike_times, event_spike_matrix = event_related_responses(self.times,
                                                                        self.clusters,
                                                                        event_times,
                                                                        duration=duration,
                                                                        dt=dt,
                                                                        nclusters=self.nclusters)

        return event_spike_times, event_spike_matrix

    def spike_count_matrix(self, dt=0.001, t0=0, duration=None, dtype=None):
        '''Generate a spike count matrix for all clusters.

        Parameters
        ----------
        dt : float-like
            Bin size in [secs]
        t0 : float-like
            Time from which to start counting spikes in [secs]
            Defaults to 0. If `t0 is None`, then t0 is defined
            by the onset of the first bin containing a spike.
        duration : float-like
            Number of [seconds] after `t0` to count spikes up to
        dtype : optional
            Defaults to using a small unsigned integer datatype

        Returns
        -------
        newtimes     : 1D np.ndarray (n,)
            Bin onset times for the spike matrix
        spike_matrix : 2D np.ndarray (n, m)
            Matrix containing spike counts for all `m` clusters.
        '''
        newtimes, spike_matrix = spike_count_matrix_from_sparse(
            self.times, self.sparse_matrix,
            dt=dt, t0=t0, duration=duration, dtype=dtype)
        return newtimes, spike_matrix

    def time_locked_spike_matrix(self, times, dt=0.010):
        '''Resample spikes at the requested times with a fixed spike window width.

        Spikes are counter from onset to onset+dt (pythonic: left inclusive, right inclusive).

        Parameters
        ----------
        times : np.ndarray (nsamples,)
            Event of onsets requested
        dt : float-like
            Window after onset to count spikes upto.

        Returns
        -------
        spike_matrix : np.ndarray (nsamples, nclusters)
        '''
        spike_matrix = time_locked_spike_matrix(self.times, self.clusters,
                                                times, dt,
                                                nclusters=self.nclusters)
        return spike_matrix

    def time_locked_mua_matrix(self, times, dt, cluster_depths=None, good_clusters=None, **kwargs):
        '''Compute MUA matrix at the requested times with a fixed spike window width.

        Parameters
        ----------
        times : np.ndarray (nsamples,)
            Event of onsets requested
        dt : float-like
            Window after onset to count spikes upto.
        cluster_depths : np.ndarray (nclusters,)
            Depth of each spike cluster
        good_clusters : np.ndarray (nclusters,)
            Selects clusters to use in MUA computation.
        **kwargs : dict
            Passed onto ``mua_from_spike_cluster_matrix()``.
            For example, ``chunk_sizeum=200``

        Returns
        -------
        mua_nclusters : np.ndarray (nmua,)
            Number of clusters in each MUA chunk
        mua_matrix : np.ndarray (nsamples, nmua)
            MUA matrix
        '''
        spike_matrix = self.time_locked_spike_matrix(times, dt=dt)

        if cluster_depths is None:
            cluster_depths = self.cluster_depths

        if good_clusters is None:
            good_clusters = self.good_clusters
        mua_nclusters, mua_matrix = mua_from_spike_cluster_matrix(spike_matrix, cluster_depths, good_clusters=good_clusters, **kwargs)
        return mua_nclusters, mua_matrix
