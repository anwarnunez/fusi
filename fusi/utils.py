import time
import warnings
import numpy as np
from scipy import signal
from tikreg import models, utils as tikutils

from tikreg import models
from tikreg import spatial_priors, temporal_priors

from fusi.config import DATA_ROOT


def get_peak_info(time, data, sumfunc=np.median):
    '''
    time : 1D np.array
    data : 1D np.array
    '''
    assert data.ndim == 1
    data = np.atleast_2d(data)

    peak_location = np.argmax(sumfunc(data, 0))
    peak_time = time[peak_location]
    peak_width = signal.peak_widths(sumfunc(data, 0), [peak_location], rel_height=0.5)[0]
    peak_fwhm = peak_width*0.3 # fUSI dt = 0.3 sec
    return peak_time, float(peak_fwhm)

def summarize_peaks(time, data, **kwargs):
    '''
    '''
    peak_time, peak_fwhm = np.asarray([get_peak_info(time, dat, **kwargs) for dat in data]).T
    return peak_time, peak_fwhm


def MAD(data, axis=0):
    '''
    '''
    med = np.median(data, axis=axis, keepdims=True)
    mad = np.median(np.abs(data - med), axis=axis)
    return mad


def pct_signal_change(data):
    pct = ((data/data.mean(0)) - 1.0)*100.0
    return pct


def glob_loader(pattern, **kwargs):
    '''
    subject='CR017',
    session='2019-11-13',
    stimulus_condition='kalatsky',
    block_numbers=None,
    results_version='results_v03',
    data_root=DATA_ROOT
    '''
    from fusi.extras import readers
    for fl, key in glob_local(pattern, **kwargs):
        yield readers.hdf_load(fl, key)


def load_fusinpx_results(prefix='regression',
                         data_pattern='halfidx-1*weights',
                         time_pattern='halfidx-1*delaysec',
                         area_name='V1',
                         probe_names=['probe00', 'probe01'],
                         res_transform=lambda x: x,
                         flatten=False,
                         verbose=False,
                         **kwargs):
    '''
    Parameters
    ----------
    probe_names : str or None
    time_pattern : str or none
    flatten : bool
        True: Useful when arrays are of different size, (dataset*samples)
        False: Output shape is (dataset, nsamples)
    res_transform : function
        Function applied to each dataset after squeezing.

    kwargs : dict
        Passed to glob_pattern() for finer selection:
        * subject='*'
        * session='*'
        * stimulus_condition='*'
        * block_numbers=None
        * results_version='results_v03'
        * exclude_patterns=[('2019-11-26', None)],

    Available patterns
    ------------------
    `prefix`, followed by relelevant key in `data_pattern`

    * shuffcoh: {freq, coherence_boot, coherece_median, coherence_mean}
    * slicesview: {mean,std,tsnr}_methodC.hdf
    * timecourses: frunits{spksec,zscore,pctchg}_{probe00,probe01}_{fusi,ephys}
    * regression: {probe00,probe01}_{halfidx0,halfidx-1}_fitl2_{performance,predictions,weights,Ytest}_methodC.hdf
    * crosscorrelation: {probe00, probe00}_{delaysecs,tpcorr}_methodC.hdf
    * coherence: {probe00,probe01}_{freq,coherence}_methodC.hdf

    subject='CR017',
    session='2019-11-13',
    stimulus_condition='kalatsky',
    block_numbers=None,
    results_version='results_v03',
    data_root=DATA_ROOT
    '''
    results = []
    time = None

    if probe_names is None:
        # future-proofing
        probe_name = ''
        raise ValueError('Untested for non-probe data')

    for probe_name in probe_names:
        matcher = f'*{prefix}*{probe_name}*{data_pattern}*'
        print(matcher, kwargs)
        res = list(glob_pattern(matcher,
                                probe_name=probe_name,
                                area_name=area_name,
                                verbose=verbose,
                                do_loading=True, **kwargs))


        if len(res) > 0:
            res = [res_transform(t.squeeze()) for t in res]
            res = np.asarray(res) if flatten is False else np.hstack(res)
            results.append(res.squeeze())

            if time_pattern is not None and time is None:
                # Only load time information once
                time_matcher = f'*{prefix}*{probe_name}*{time_pattern}*'
                print(time_matcher)
                time = np.asarray(list(glob_pattern(
                    time_matcher,
                    probe_name=probe_name,
                    area_name=area_name,
                    verbose=verbose,
                    do_loading=True,
                    **kwargs)))

                if len(time) == 0:
                    raise ValueError('No time info found!')

    if len(results) == 0:
        print('Nothing found: %s'%matcher, kwargs)
        return None, None

    if len(results) == 1:
        results = results[0]
    else:
        results = np.vstack(results) if flatten is False else np.hstack(results)


    # time-vector is the same for all
    if time is not None:
        assert time.shape[-1] == results.shape[-1]
        time = time[0]
        print('shapes: times, data:', time.shape, results.shape)
    else:
        print('shapes:', results.shape)

    return time, results


def glob_pattern(pattern,
                 subject='*',
                 session='*',
                 stimulus_condition='*',
                 block_numbers=None,
                 results_version='results_v03',
                 exclude_patterns=None, #[('2019-11-26', None)],
                 data_root=DATA_ROOT,
                 verbose=False,
                 probe_name=None,
                 area_name=None,
                 do_loading=False):
    '''
    Patterns
    --------
    * slicesview: {mean,std,tsnr}_methodC.hdf
    * timecourses: frunits{spksec,zscore,pctchg}_{probe00,probe01}_{fusi,ephys}
    * regression: {probe00,probe01}_{halfidx0,halfidx-1}_fitl2_{performance,predictions,weights,Ytest}_methodC.hdf
    * crosscorrelation: {probe00, probe00}_{delaysecs,tpcorr}_methodC.hdf
    * coherence: {probe00,probe01}_{freq,coherence}_methodC.hdf

    subject='CR017',
    session='2019-11-13',
    stimulus_condition='kalatsky',
    block_numbers=None,
    results_version='results_v03',
    data_root=DATA_ROOT
    '''
    from fusi.extras import readers
    from fusi import handler2 as handler
    EXPERIMENTS = {'CR017' : ['2019-11-13','2019-11-14'],
                   'CR020' : ['2019-11-20','2019-11-21', '2019-11-22'],
                   'CR019' : ['2019-11-26','2019-11-27'],
                   'CR022' : ['2020-10-07','2020-10-11'],
                   'CR024' : ['2020-10-29']}


    if stimulus_condition != '*':
        assert stimulus_condition in ['kalatsky', 'checkerboard', 'spontaneous']

    if subject=='*':
        # Use all subjects
        pass
    elif isinstance(subject, (list, str)):
        # select some subjects
        EXPERIMENTS = {k : v for k,v in EXPERIMENTS.items() if k in subject}

    for sub, sessions in EXPERIMENTS.items():
        if session=='*':
            # Use all sesssions
            pass
        else:
            # select requested sessions only
            sessions = [sess for sess in sessions if sess in session]

        # iterate thru sessions
        for sess in sessions:
            matches = glob_local(pattern,
                                 subject=sub,
                                 session=sess,
                                 stimulus_condition=stimulus_condition,
                                 block_numbers=block_numbers,
                                 results_version=results_version,
                                 probe_name=probe_name,
                                 area_name=area_name,
                                 exclude_patterns=exclude_patterns,
                                 verbose=verbose,
                                 data_root=data_root)

            # iterate thru matches
            for (fl, key) in matches:
                if do_loading:
                    if verbose: print('%s : %s'%(fl, key), sub, sess, stimulus_condition)
                    data = readers.hdf_load(fl, key, verbose=False)
                    if np.allclose(data, 0):
                        print('Data is all zeros:',fl, key)
                    yield data
                else:
                    yield fl, key


def glob_local(pattern,
               subject='CR017',
               session='2019-11-13',
               stimulus_condition='kalatsky',
               block_numbers=None,
               results_version='results_v03',
               probe_name=None,
               area_name=None,
               exclude_patterns=None,
               verbose=False,
               analysis_valid_only=True,
               data_root=DATA_ROOT):
    '''
    valid_only : bool
        Use only data defined as good from the subject LOG.INI

    Example
    -------
    >>> pattern = '*frunitzscore*probe00*ephys*'
    >>> files = glob_local(pattern)
    '''
    from pathlib import Path
    from fusi.extras import readers
    from fusi import handler2 as handler
    path = Path(data_root).joinpath(subject, session, results_version)
    files = path.glob(pattern)

    for fl in  files:
        contents = readers.hdf_load(fl, verbose=False)
        block_mapper = {int(t.split('_')[1].strip('block')) : t for t in contents}
        if block_numbers is not None:
            # In case we want to restrict to a subset of blocks
            block_mapper = [b for b in block_mapper if b in block_numbers]

        if analysis_valid_only:
            block_mapper = {k : v  for k,v in block_mapper.items() if k in handler.MetaSession(subject, session, verbose=False).analysis_blocks}

        if stimulus_condition is not None and stimulus_condition!='*':
            block_mapper = {blk_number: key for idx,(blk_number,key) in enumerate(block_mapper.items()) \
                if handler.MetaBlock(subject, session, blk_number, verbose=False).task_name==stimulus_condition}

        if area_name is not None and area_name != '*':
            assert probe_name is not None
            block_mapper = {blk_number: key for idx,(blk_number,key) in enumerate(block_mapper.items()) \
                if (handler.MetaBlock(subject, session, blk_number,verbose=False).fusi_get_probe_allen_mask(
                        probe_name, allen_area_name=area_name) is not None)}

        for block_number, block_key in sorted(block_mapper.items()):
            if exclude_patterns is not None:
                is_valid = True
                for exclude_fl, exclude_key in exclude_patterns:
                    if (exclude_fl in str(fl)) and (str(exclude_key) in block_key):
                        if verbose: print('Skipping:', exclude_fl, exclude_key)
                        is_valid = False
                        continue
                if is_valid:
                    yield fl, block_key
            else:
                yield fl, block_key




def roiwise_pctsignalchange(data,
                            population_mean=False,
                            clip_value=75,
                            trim_beg=0, trim_end=None):
    '''data is time by units

    pct signal change per pixel, then mean across pixels
    '''
    data = data[trim_beg:trim_end].mean(-1, keepdims=True)
    data /= data.mean(0)
    data -= 1.0
    data *= 100

    # clip this
    data = np.clip(data, -75, 75)
    return data

def pixelwise_pctsignalchange_median(data, **kwargs):
    return pixelwise_pctsignalchange(data, normfunc=np.median, **kwargs)

def pixelwise_pctsignalchange(data,
                              population_mean=False,
                              clip_value=None,
                              clip_sd=None,
                              normfunc=np.mean,
                              trim_beg=0, trim_end=None):
    '''Compute the percent signal change per unit.

    Parameters
    ----------
    data : np.ndarray, (ntimepoints, ..., nunits)
    clip_value : scalar
        Threshold for % signal changes
    population_mean : bool
        Mean across last dimension
    trim_beg, trim_end:
        Ignore these samples when computing % signal change

    Returns
    -------
    pct_signal_change : np.ndarray, (ntimepoints, ..., nunits)
        Signals in units of % (ie 10% not 0.1) in the
        range (-clip_value, clip_value), mean=0.
    '''
    data /= normfunc(data[trim_beg:trim_end], axis=0)
    data -= 1.0
    data *= 100
    # clip this

    sdvals = data.std(0)

    if clip_value is not None:
        data = np.clip(data, -clip_value, clip_value)
    if clip_sd is not None:
        data = np.clip(data, -(sdvals*clip_sd), sdvals*clip_sd)

    if population_mean:
        data = data.mean(-1, keepdims=True)
    return data


def normalize_fusi(data_matrix):
    '''
    '''
    return pct_signal_change(np.sqrt(data_matrix))


def fast_find_between(continuous_times, discrete_times, window=0.050):
    '''A parallel implementation of::

    markers = np.logical_and(continuous_units >= discrete_units[0],
                             continuous_units <  discrete_units[0] + window)

    for all elements of `discrete_units`.

    Parameters
    ----------
    continuous_units (1D np.ndarray) : (n,) vector of unit sampling (e.g. secs, mm]
    discrete_units (1D np.ndarray)   : (m,) Vector of new unit-stamps
    window (scalar)                  : Window of unit after unit (e.g. secs, mm]

    Returns
    -------
    markers (list of `m` arrays) :
        Each element of the list contains the mask (n,)

    Examples
    --------
    >>> fast_ratehz, new_ratehz = 100, 10
    >>> times = np.arange(10000)/fast_ratehz + np.random.rand(10000)/fast_ratehz
    >>> new_times = np.linspace(0, times.max(), len(times)/(fast_ratehz/new_ratehz))
    >>> markers = fast_find_times_between(times, new_times, window=1./new_ratehz)
    '''
    from fusi import mp
    def find_between(vec, before, window=0.050):
        '''
        '''
        matches = np.logical_and(vec >= before, vec < before+window)
        matches = matches.nonzero()[0]
        # # Same speed
        # matches = (vec >= before)*(vec < before+window).nonzero()[0]
        # # Same speed-ish
        # greater = (vec >= before).nonzero()[0]
        # matches = greater[vec[greater] < (before + window)]
        return matches

    func = lambda x: find_between(continuous_times, x, window=window)
    res = mp.map(func, discrete_times, procs=10)
    return res


def spikes_at_times(event_times, post_event_time,
                    spike_times, spike_clusters,
                    nclusters=None):
    '''
    '''
    from fusi.mp import map as parallelmap
    from scipy import sparse

    if nclusters is None:
        # use highest cluster index
        nclusters = spike_clusters.max() + 1

    shape = (len(spike_times), nclusters)
    sparse_spike_matrix = sparse.csr_matrix((np.ones_like(spike_times),
                                             (np.arange(len(spike_times)), spike_clusters)),
                                            shape, dtype=np.bool)
    spike_bins = fast_find_between(spike_times, event_times, window=post_event_time)

    func = lambda x: np.asarray(sparse_spike_matrix[x]) # spike counter function
    spike_matrix = np.asarray(parallelmap(func, spike_bins, procs=10)).squeeze()

    return spike_matrix


def sparse_spike_matrix(spike_times, spike_clusters, nclusters=None):
    '''
    '''
    from scipy import sparse

    if nclusters is None:
        # use highest cluster index
        nclusters = spike_clusters.max() + 1

    shape = (len(spike_times), nclusters)
    sparse_spike_matrix = sparse.csr_matrix((np.ones_like(spike_times),
                                             (np.arange(len(spike_times)), spike_clusters)),
                                             shape, dtype=np.bool)
    return sparse_spike_matrix


def bin_spikes(new_onsets, window,
               spike_times, spike_clusters,
               nclusters=None):
    '''Construct a spike counts matrix at the temporal resolution requested.

    Parameters
    ----------
    new_onsets : 1D np.ndarray (m,)
        Times at which to start counting spikes [seconds]
    window : scalar
        Bin size in [seconds]
    spike_times : 1D np.ndarray (n,)
        Vector of spike time-stamps in [seconds]
    spike_clusters : 1D np.ndarray (n,)
        Vector of cluster indexes
    nclusters (optional) : scalar
        total number of clusters `k`.
        if not given, `nclusters = spike_clusters.max() + 1`

    Returns
    -------
    spike_matrix : 2D np.ndarray (m, k)
    '''
    from fusi.io import spikes
    spike_matrix = spikes.time_locked_spike_matrix(spike_times,
                                                   spike_clusters,
                                                   new_onsets,
                                                   window,
                                                   nclusters=nclusters)
    ##########################
    # This is ~10x slower:
    ##########################
    #
    # from fusi.mp import map as parallelmap
    # from scipy import sparse
    #
    # if nclusters is None:
    #     # use highest cluster index
    #     nclusters = spike_clusters.max() + 1
    #
    # shape = (len(spike_times), nclusters)
    # sparse_spike_matrix = sparse.csr_matrix((np.ones_like(spike_times),
    #                                          (np.arange(len(spike_times)), spike_clusters)),
    #                                         shape, dtype=np.bool)
    # spike_bins = fast_find_between(spike_times, new_onsets, window=window)
    #
    # func = lambda x: np.asarray(sparse_spike_matrix[x].sum(0)) # spike counter function
    # spike_matrix = np.asarray(parallelmap(func, spike_bins, procs=10)).squeeze()
    return spike_matrix


def bin_spikes_turbo(window,
                     spike_times, spike_clusters,
                     nclusters=None,
                     first_bin_onset=0.0):
    '''Construct a spike counts matrix at the temporal resolution requested.

    Parameters
    ----------
    window : scalar
        Bin size in [seconds]
    spike_times : 1D np.ndarray (n,)
        Vector of spike time-stamps in [seconds]
    spike_clusters : 1D np.ndarray (n,)
        Vector of cluster indexes
    nclusters (optional) : scalar
        total number of clusters `k`.
        if not given, `nclusters = spike_clusters.max() + 1`
    first_bin_onset : scalar-like
        Onset of first bin [seconds]

    Returns
    -------
    spike_matrix : 2D np.ndarray (m, k)
    '''
    from fusi.mp import map as parallelmap
    from scipy import sparse

    if nclusters is None:
        # use highest cluster index
        nclusters = spike_clusters.max() + 1

    shape = (len(spike_times), nclusters)
    sparse_spike_matrix = sparse.csr_matrix((np.ones_like(spike_times),
                                             (np.arange(len(spike_times)), spike_clusters)),
                                            shape, dtype=np.bool)
    spike_bins = (spike_times - first_bin_onset) // window
    nbins = int(np.ceil(spike_bins.max())) + 1
    unique_bins = np.unique(spike_bins)
    spike_matrix = np.zeros((nbins, nclusters), dtype=np.int32)

    func = lambda uqbin: np.asarray(sparse_spike_matrix[spike_bins==uqbin,:].sum(0)).squeeze()
    spike_counts = parallelmap(func, unique_bins)
    for idx, scount in zip(unique_bins, spike_counts):
        spike_matrix[int(idx)] = scount
    matrix_times = np.arange(nbins)*window + first_bin_onset
    return matrix_times, spike_matrix


class Bunch(dict):
    """A subclass of dictionary with an additional dot syntax."""
    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def copy(self):
        """Return a new Bunch instance which is a copy of the current Bunch instance."""
        return Bunch(super(Bunch, self).copy())


def mk_log_fusi(raw_mean_image):
    img = raw_mean_image.copy()
    img -= img.min()
    img /= img.max()
    img *= 1000
    img = np.log(img + 1)
    return img

def brain_mask(raw_mean_image):
    '''
    '''
    from skimage.filters import threshold_otsu


    thresh  = threshold_otsu(img)
    binary = img > thresh
    return binary


def cvridge_nodelays(Xfull, Yfull,
                     Xtest=None,
                     Ytest=None,
                     ridges=np.logspace(0,3,10), chunklen=10,
                     metric='rsquared', population_optimal=False, delays_mean=True,
                     performance=True, weights=False, predictions=False, verbose=2):
    '''
    '''
    if Xtest is None and Ytest is None:
        ntrain = int(Xfull.shape[0]/2)
        Xtrain, Xtest = Xfull[:ntrain], Xfull[ntrain:]
        Ytrain, Ytest = Yfull[:ntrain], Yfull[ntrain:]
    else:
        Xtrain, Ytrain = Xfull, Yfull

    nfeatures = Xtrain.shape[1]
    feature_prior = spatial_priors.SphericalPrior(nfeatures)
    temporal_prior = temporal_priors.SphericalPrior(delays=[0])

    fit_spherical = models.estimate_stem_wmvnp([Xtrain], Ytrain,
                                               [Xtest],Ytest,
                                               feature_priors=[feature_prior],
                                               temporal_prior=temporal_prior,
                                               ridges=ridges,
                                               folds=(2,5),
                                               verbosity=verbose,
                                               performance=performance,
                                               weights=weights,
                                               predictions=predictions,
                                               population_optimal=population_optimal, # pop opt is more stable
                                               metric=metric,
                                               chunklen=chunklen) # correlation is more stable

    if weights is True:
        kernel_weights = np.nan_to_num(fit_spherical['weights'])
        voxels_ridge = fit_spherical['optima'][:, -1]
        weights_mean = models.dual2primal_weights_banded(kernel_weights,
                                                         Xtrain,
                                                         np.ones_like(voxels_ridge),
                                                         temporal_prior,
                                                         delays_mean=delays_mean,
                                                         verbose=True)
        fit_spherical['weights'] = weights_mean
    return fit_spherical



def generate_bootstrap_samples(data_size, subsample_size=100,
                               chunk_size=20, nresamples=1000):
    '''sample 1000 sequences `s` long from the full dataset `data_size`


    data_size : int
        n-samples in the full dataset
    subsample_size : int
        sample `s` samples from the full dataset
    chunks_size : int
        Break data into non-overlapping chunks
        of this size before bootstrapping samples
    nresamples : int
        Number of bootstrap resamples
    '''
    assert data_size < 2**16    # this will be soooo slow
    dtype = np.uint16

    nsplits = int(data_size / chunk_size)
    nchunks = int(subsample_size / chunk_size)
    print(nsplits, nchunks)
    for idx in range(nresamples):
        start = np.random.randint(0, chunk_size, dtype=dtype)
        indices = np.arange(start, data_size, dtype=dtype)
        splits = np.array_split(indices, nsplits)
        # gidx = sorted(np.random.permutation(nsplits)[:nchunks]) # w/o replacement
        gidx = np.random.randint(0, nsplits, size=nchunks) # w/ replacement
        sample = np.hstack([splits[t] for t in gidx])
        yield sample



def bootstrap_ols_fit(xtrain, ytrain, bootsamples,
                      include_constant=False):
    '''bootstrap the bhat estimate with OLS
    '''
    nboot = len(bootsamples)
    if include_constant:
        xtrain = np.hstack([np.ones((xtrain.shape[0], 1), dtype=xtrain.dtype),
                            xtrain])
    boot_weights = np.zeros((nboot, xtrain.shape[1], ytrain.shape[1]), dtype=xtrain.dtype)
    for bdx, samplesidx in enumerate(bootsamples):
        bhat = models.ols(xtrain[samplesidx], ytrain[samplesidx])
        boot_weights[bdx] = bhat
        if bdx % 500 == 0:
            print(bdx, bhat.shape)
    return boot_weights


def bootstrap_sta_fit(xtrain, ytrain, bootsamples,
                      include_constant=False):
    '''bootstrap the bhat estimate with OLS
    '''
    nboot = len(bootsamples)
    if include_constant:
        xtrain = np.hstack([np.ones((xtrain.shape[0], 1), dtype=xtrain.dtype),
                            xtrain])
    boot_weights = np.zeros((nboot, xtrain.shape[1], ytrain.shape[1]), dtype=xtrain.dtype)
    for bdx, samplesidx in enumerate(bootsamples):
        bhat = np.dot(xtrain[samplesidx].T, ytrain[samplesidx])/len(samplesidx)
        boot_weights[bdx] = bhat
        if bdx % 500 == 0:
            print(bdx, bhat.shape)
    return boot_weights



def bootstrap_ridge_fit(xtrain, ytrain, bootsamples,
                        include_constant=False,
                        ridges=np.logspace(0,4,10)):
    '''bootstrap the bhat estimate with OLS
    '''
    nboot = len(bootsamples)
    if include_constant:
        xtrain = np.hstack([np.ones((xtrain.shape[0], 1), dtype=xtrain.dtype),
                            xtrain])
    boot_weights = np.zeros((nboot, xtrain.shape[1], ytrain.shape[1]), dtype=xtrain.dtype)
    solver = models.solve_l2_primal if xtrain.shape[0] > xtrain.shape[1] \
             else models.solve_l2_dual

    for bdx, samplesidx in enumerate(bootsamples):
        fit = solver(xtrain[samplesidx], ytrain[samplesidx],
                     xtrain[samplesidx], ytrain[samplesidx],
                     performance=True,
                     ridges=ridges,
                     metric='rsquared',
                     verbose=True if bdx % 1000 == 0 else False,
                     )
        # population best
        perf = np.nanmean(fit['performance'], -1)
        ridgeperf, ridgeopt = np.max(perf), ridges[np.argmax(perf)]

        if np.isnan(ridgeperf):
            ridgeopt = np.inf

        fit = solver(xtrain[samplesidx], ytrain[samplesidx],
                     xtrain[samplesidx], ytrain[samplesidx],
                     weights=True,
                     ridges=[ridgeopt],
                     metric='rsquared')

        #                      ridges=np.logspace(0,4,10),
        # fit = models.cvridge(xtrain[samplesidx], ytrain[samplesidx],
        #                      ridges=np.logspace(0,4,10),
        #                      weights=True,
        #                      metric='rsquared',
        #                      blocklen=40,
        #                      verbose=True if bdx % 1000 == 0 else False,
        #                      )

        bhat = fit['weights']
        boot_weights[bdx] = bhat
        if bdx % 500 == 0:
            print(bdx, bhat.shape)
    return boot_weights


def detrend_secs2oddwindow(secs, dt):
    '''Valid window length for detrending

    Parameters
    ----------
    secs : scalar, [seconds]
    dt : scalar, [seconds]
    '''
    window_size = int(secs/dt)
    window_size += (window_size - 1) % 2
    return window_size


def detrend_signal(data,
                   window_size=5,
                   detrend_type='median',
                   **kwargs):
    '''Detrend a signal

    Parameters
    ----------
    data : np.ndarray, (ntimepoints, ...)
    window_size : scalar, (k,)
    detrend_type : str
        * median: Median filter
        * mean: Moving average filter
        * macausal: Causal moving average filter
        * bandpass: Bandpass signal
        * sg: savitsky-golay filter
    **kwargs : for sg detrending

    Returns
    -------
    detrended_data : np.ndarray, (ntimepoints, ...)
    '''
    shape = data.shape
    data = data.reshape(data.shape[0], -1)
    detdat = temporal_filter(
        data, window_size=window_size, ftype=detrend_type, **kwargs)

    detdat = detdat.reshape(shape)
    return detdat


def temporal_filter(fusi, window_size=5,
                    ftype='median', **fkwargs):
    '''Estimate low-frequency trend and removes it from signal

    Parameters
    ----------
    fusi (ntimepoints, voxels): np.ndarray
    '''
    assert ftype in ['median', 'sg', 'ma', 'macausal','bandpass']
    if fusi.ndim == 1:
        fusi = fusi[...,None]
    fusi_filt = np.zeros_like(fusi)
    for chidx in range(fusi.shape[1]):
        pixel_data = fusi[:, chidx] #.copy()
        pixel_mean = pixel_data.mean()
        pixel_data = pixel_data - pixel_mean
        if ftype == 'median':
            temporal_filter = signal.medfilt(pixel_data, window_size)
        elif ftype == 'sg':
            polyorder = 3 if 'polyorder' not in fkwargs else fkwargs.pop('polyorder')
            temporal_filter = signal.savgol_filter(pixel_data,
                                                   window_size,
                                                   polyorder,
                                                   **fkwargs)
        elif ftype == 'ma':
            delays = range(-window_size, window_size+1)
            temporal_filter = tikutils.delay_signal(pixel_data, delays).mean(-1)
        elif ftype == 'macausal':
            delays = range(window_size+1)
            temporal_filter = tikutils.delay_signal(pixel_data, delays).mean(-1)
        elif ftype == 'bandpass':
            temporal_filter = band_pass_signal(pixel_data, **kwargs)

        # add mean back
        new_signal = (pixel_data - temporal_filter) + pixel_mean
        fusi_filt[:, chidx] = new_signal
        if chidx % 1000 == 0:
            print(ftype, window_size, chidx, new_signal.sum(), pixel_mean, fkwargs)
    return fusi_filt


def get_centered_samples(data, outsamples):
    '''Extract "N" samples from the middle of the data.
    '''
    oldsamples = data.shape[0]
    window_left = int((oldsamples - outsamples)/2)
    #window_left = int((oldsamples - window_center)/2)

    newdata = data[window_left:window_left + outsamples]
    assert newdata.shape[0] == outsamples
    return newdata



def get_temporal_filters(fusi, window_size=5,
                         ftype='median', **fkwargs):
    '''fusi (ntimepoints, pixels)

    estimates and returns a filter
    '''
    assert ftype in ['median', 'sg', 'ma', 'macausal','bandpass']
    if fusi.ndim == 1:
        fusi = fusi[...,None]
    fusi_filt = np.zeros_like(fusi)
    for chidx in range(fusi.shape[1]):
        pixel_data = fusi[:, chidx].copy()
        pixel_mean = pixel_data.mean()
        pixel_data = pixel_data - pixel_mean
        if ftype == 'median':
            temporal_filter = signal.medfilt(pixel_data, window_size)
        elif ftype == 'sg':
            polyorder = 3 if 'polyorder' not in fkwargs else fkwargs.pop('polyorder')
            temporal_filter = signal.savgol_filter(pixel_data,
                                                   window_size,
                                                   polyorder,
                                                   **fkwargs)
        elif ftype == 'ma':
            delays = range(-window_size, window_size+1)
            temporal_filter = tikutils.delay_signal(pixel_data, delays).mean(-1)
        elif ftype == 'macausal':
            delays = range(window_size+1)
            temporal_filter = tikutils.delay_signal(pixel_data, delays).mean(-1)
        elif ftype == 'bandpass':
            temporal_filter = band_pass_signal(pixel_data, **fkwargs)

        # add mean back
        new_signal = temporal_filter + pixel_mean
        fusi_filt[:, chidx] = new_signal
        if chidx % 1000 == 0:
            print(ftype, window_size, chidx, new_signal.sum(), pixel_mean, fkwargs)
    return fusi_filt.squeeze()


def autocorr(y, yy=None, nlags=1):
    '''
    Auto-correlation function for `y` a 1D-array
    `nlags` is in sample units.

    Parameters
    ----------
    y : 1D np.ndarray
        The signal
    yy : optional
    nlags : int
        Number of lags to return (0 - N)

    Returns
    -------
    rho : 1D np.ndarray
        The auto-correlation values upto `nlags` including zero

    '''
    assert y.ndim == 1
    if yy is None:
        yy = y
    assert yy.ndim == 1

    acorr = np.correlate(y, yy, 'same')
    acorr /= acorr.max() # normalized
    n = int(acorr.shape[0]/2)
    # lags = np.arange(acorr.shape[0]) - n
    # return acorr[np.where(np.logical_and(lags >= 0, lags <= nlags))]
    # Minutely faster
    return acorr[n:n+nlags+1]


def make_mask(superior_top, superior_bottom,
              coronal_medial, coronal_lateral,
              shape):

    mask = np.zeros(shape, dtype=np.bool)
    mask[min(coronal_medial, coronal_lateral):max(coronal_medial, coronal_lateral),
         min(superior_bottom, superior_top):max(superior_bottom, superior_top)] = True
    return mask


def band_pass_signal(data, sample_rate=1.0, ntaps=5, cutoff=[24.0, 48.0]):
    '''Band-pass a signal with a FIR filter.

    data (np.ndarray)       :
    sample_rate (float-like): in Hz
    cutoff (list-like)      : (low, high) frequency cutoffs
    '''
    # filter photo signal
    from scipy import signal
    ntaps = max(int(sample_rate/float(ntaps)), 1)
    cutoff = np.asarray(cutoff)
    wn = cutoff/(sample_rate/2.0)
    print(ntaps, wn)
    fircoef = signal.firwin(ntaps, wn, window='hamming', pass_zero=False)
    fircoef = fircoef.astype(data.dtype)
    fsignal = signal.filtfilt(fircoef, 1.0, data, axis=0)
    return fsignal




def mean_angle(ffts, metric=np.mean):
    '''
    '''
    degrees = 0.0
    ffts = np.asarray(ffts).mean(0)

    for fft in ffts:
        phase =  metric(np.angle(fft), 0)
        # map -pi:pi to 0:2pi range
        phase = np.where(phase < 0, 2.0*np.pi + phase, phase)
        degrees += np.rad2deg(phase)/len(ffts)
    return degrees


class StopWatch(object):
    '''Duration lapsed since last call
    '''
    def __init__(self, verbose=True):
        self.start = time.time()
        self.verbose = verbose
    def __call__(self):
        now = time.time()
        dur = now - self.start
        self.start = now
        if self.verbose:
            print('Duration: %0.2f[sec]:'%dur)
        return dur


def hex2rgba(h):
    '''Convert a hex color into RGB
    '''
    rgb = list(int(h[i:i+2], 16) for i in (0, 2, 4))
    return tuple(rgb + [0])


chronometer = StopWatch()

def tiff2arr(fl, verbose=True):
    '''Convert a TIFF image to an array
    '''
    from PIL import Image
    with Image.open(fl, 'r') as im:
        frame = 0
        data = []
        while True:
            try:
                if verbose:
                    print(frame)
                im.seek(frame)
                arr = np.asarray(im)
                data.append(arr)
                frame += 1

            except EOFError:
                if verbose:
                    print('\nFile "%s" has %i frames' % (fl, frame))
                data = np.asarray(data)
                break
            except KeyError:
                data = None
                break
            im.close()
            return data


def lanczosfun(cutoff, t, window=3):
    """Compute the lanczos function with some cutoff frequency [B]Hz at some time [t].
    [t] can be a scalar or any shaped numpy array.
    If given a [window], only the lowest-order [window] lobes of the sinc function
    will be non-zero.
    """
    t = np.atleast_1d(t * cutoff)
    val = window * np.sin(np.pi*t) * np.sin(np.pi*t/window) / (np.pi**2 * t**2)
    val[t==0] = 1.0
    val[np.abs(t)>window] = 0.0
    return val


def lanczosinterp2D(data, oldtime, newtime, window=3, cutoff=1.0, rectify=False):
    """Interpolates the columns of [data], assuming that the i'th row of data corresponds to
    oldtime(i). A new matrix with the same number of columns and a number of rows given
    by the length of [newtime] is returned.

    [cutoff] is the frequency cutoff in Hz. This defines the first zero-crossing of the lanczos filter

    The time points in [newtime] are assumed to be evenly spaced, and their frequency will
    be used to calculate the low-pass cutoff of the interpolation filter.

    [window] lobes of the sinc function will be used. [window] should be an integer.
    """
    ## Find the cutoff frequency ##
    #cutoff = 1/np.mean(np.diff(newtime)) * cutoff_mult
    print ("Doing lanczos interpolation with cutoff=%0.3f and %d lobes." % (cutoff, window))

    ## Build up sinc matrix ##
    sincmat = np.zeros((len(newtime), len(oldtime)), dtype=np.float32)
    for ndi in range(len(newtime)):
        sincmat[ndi,:] = lanczosfun(cutoff, newtime[ndi]-oldtime, window)

    if rectify:
        newdata = np.hstack([np.dot(sincmat, np.clip(data, -np.inf, 0)),
                             np.dot(sincmat, np.clip(data, 0, np.inf))])
    else:
        ## Construct new signal by multiplying the sinc matrix by the data ##
        newdata = np.dot(sincmat, data)

    return newdata
