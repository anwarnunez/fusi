import numpy as np
from scipy.stats import zscore

from tikreg import utils as tikutils
from fusi import handler2 as handler, utils as futils
from fusi.io import spikes

from fusi.extras import readers


##############################
# Globals
##############################
zs = zscore

fusi_options = {'300': dict(dt_ms=300, window_ms=400, svddrop=15, freq_cutoffhz=15)}

signal_units_normalization = {'pixpctchg': futils.pixelwise_pctsignalchange,
                              # 'pixpctchg' : futils.pixelwise_pctsignalchange_median,
                              'roipctchg': futils.roiwise_pctsignalchange,
                              'zscore': zs,
                              }

SPIKE_METHOD = 'methodC'
AREA_NAME = 'HPC'
PROBE_ROI_NAMES = {'HPC': 'X1',
                   'V1': 'V1'}
FUSI_PROBE_MASK_NAME = {'HPC': 'hippo',
                        'V1': 'V1'}

STIMULUS_CONDITIONS = ['spontaneous', 'kalatsky', 'checkerboard']

# Proportion of voxels traversed by NPx in this slice belonging to ROI
# [%] 0-100 range, basically a soft exact; only uses the fUSI ROI if the probe touches it
CRITERION = 1
EXACT_LOCATION = False

##############################
# helper fucntions
##############################


def get_population_hrf(area_name, condition, negation=False, normalize=True, t0=0):
    '''
    Get the mean HRF for the requested condition

    Parameters
    ----------
    area_name : str
        One of "V1" (visual cortex) or "HPC" (hippocampal formation)
    condition : str
        One of 'spontaneous', 'kalatsky' or 'checkerboard'
    negation : bool
        If True, returns the HRF computed for the other conditions excluding the one requested
    t0 : scalar
        First time point of HRF
    '''
    from fusi.config import DATA_ROOT

    assert condition in STIMULUS_CONDITIONS
    if negation:
        HRF, hrf_times = readers.hdf_load('{path}/extras/HRFs_not{stim}.hdf'.format(
            stim=condition, path=DATA_ROOT), [area_name, 'times'])
    else:
        HRF, hrf_times = readers.hdf_load('{path}/extras/HRFs_only{stim}.hdf'.format(
            stim=condition, path=DATA_ROOT), [area_name, 'times'])
    hrf_valid = hrf_times >= t0
    hrf_times = hrf_times[hrf_valid]
    HRF = HRF[:, hrf_valid]
    HRF = HRF.mean(0)

    if normalize:
        HRF /= HRF.max()
    return hrf_times, HRF


def normalize_mua_matrix(mua_matrix, spike_method='methodC', fusi_dt=0.3):
    '''Normalize the MUA matrix
    '''
    # ormalize the number of cells per MUA cluster # comment out for methodC
    if spike_method == 'methodA' or spike_method == 'methodB':
        # zscored spikes works methodsA/B
        mua_matrix = np.nan_to_num(zs(mua_spikes))
        # use the sum of MUA chunks
        total_spikes = mua_matrix.sum(-1, keepdims=True)

    elif spike_method == 'methodC':
        total_spikes = mua_matrix.mean(-1, keepdims=True)
        total_spikes *= 1./fusi_dt  # convert to spikes/second
    else:
        raise ValueError('Unknown spike normalization: %s' % spike_method)
    return total_spikes


def detrend_fusi(fusi_times, fusi_data, mask,
                 temporal_filter='none',
                 temporal_filter_window=35,
                 population_mean=True,
                 output_normalization='pixpctchg',
                 clip_value=None, clip_sdvalue=3,
                 fusi_dt=0.3,
                 verbose=False,
                 trim_before_beg=5, trim_before_end=5,
                 trim_after_beg=10, trim_after_end=10):
    '''
    times are in [secs]
    '''
    units_normalization_function = signal_units_normalization[output_normalization]

    times_valid = np.logical_and(fusi_times > (fusi_times.min() + trim_before_beg),  # trim XX seconds from start
                                 fusi_times < (fusi_times.max() - trim_after_beg))  # trim XX seconds from end
    fusi_times = fusi_times[times_valid]
    fusi_data = fusi_data[times_valid]
    fusi_shape = fusi_data.shape
    if verbose:
        print(fusi_times.shape, fusi_data.shape)

    # Extract time course for probe ROI
    if mask is not None:
        fusi_data = fusi_data[:, mask]
    fusi_data = np.nan_to_num(fusi_data)

    if temporal_filter in ['sg', 'median']:
        window_size = int(temporal_filter_window /
                          fusi_dt)        # 30 [sec] median
        window_size += (window_size - 1) % 2  # make odd number
        print(temporal_filter_window, window_size, window_size*fusi_dt)
        yfull = futils.temporal_filter(fusi_data - fusi_data.mean(0),
                                       ftype=temporal_filter,
                                       window_size=window_size, polyorder=2) + fusi_data.mean(0)
        yfull = units_normalization_function(yfull, population_mean=population_mean,
                                             clip_value=clip_value, clip_sd=clip_sdvalue)
        yfull -= yfull.mean(0)
    else:
        yfull = units_normalization_function(fusi_data.copy(), population_mean=population_mean,
                                             clip_value=clip_value, clip_sd=clip_sdvalue)

    # trim another 10 secs after detrending. Total trim is 30 secs (15 total at beg and end)
    times_valid = np.logical_and(fusi_times > (fusi_times.min() + trim_after_beg),  # trim XX seconds from start
                                 fusi_times < (fusi_times.max() - trim_after_end))  # trim XX seconds from end
    yfull = yfull[times_valid]
    fusi_times = fusi_times[times_valid]
    fusi_data = fusi_data[times_valid]
    if verbose:
        print(fusi_times.shape, fusi_data.shape)
    yfull -= yfull.mean(0)
    return fusi_times, yfull


def get_fusi_roi_mask_within_npxprobe(subject_block,
                                      probe_name,
                                      allen_area_name,
                                      criterion=CRITERION,
                                      hrf_convolve=False,
                                      fusi_mirror=False,
                                      temporal_filter='none',
                                      # slice_widthmm=0.5,
                                      temporal_filter_window=35):
    '''
    '''
    from fusi import experiments, allen
    # 2D Projection of NPx probe in fUSI
    # , 'fusi_%s_section_in_probe_roi.hdf'%FUSI_PROBE_MASK_NAME)
    probe_mask = subject_block.fusi_get_probe_master_mask(probe_name)

    brain_mask = np.logical_not(subject_block.fusi_get_outsidebrain_mask())
    pmask = np.logical_and(brain_mask, probe_mask)

    # ROI voxels in Allen Atlas
    allenccf_areas = subject_block.meta_session.fusi_get_allenccf_slices(
        subject_block.slice_number)
    area_mask = allen.mk_area_mask_from_aligned(
        allenccf_areas, allen_area_name, verbose=False)
    pmask = np.logical_and(area_mask, pmask)

    if fusi_mirror is True:
        # Mirror 2D projection for bilateral analysis
        # (This way b/c the probe track might not hit the area if mirrored first)
        pmask = pmask[:, ::-1]

    # 3D projection of NPx probe in fUSI
    # slice_position_mm = subject_block.meta_session.fusi_slice_position(subject_block.slice_number)
    slice_search_widthmm = subject_block.analysis_slicewidth_search

    fusi_probe_depth, fusi_probe_mask, probe_limits_mm = subject_block.meta_session.fusi_get_slice_probedepth_and_voxels(
        # subject_block.slice_number, probe_name, widthmm=0.8 if slice_position_mm==0 else slice_widthmm)
        subject_block.slice_number, probe_name, widthmm=slice_search_widthmm)

    # Percentage of 3D NPx probe mask inside the Allen Mask
    pct_probe_in_roi = (np.logical_and(
        fusi_probe_mask, pmask).sum()/fusi_probe_mask.sum())*100

    # Check that the probe actually touches the Allen area of interset at this slice
    if (pct_probe_in_roi < criterion) and (fusi_mirror is False):
        print('%0.02f<%0.1f%% of Neuropixels %s is in %s in slice=%i\nSkipping %s: ' % (pct_probe_in_roi,
                                                                                        criterion,
                                                                                        probe_name,
                                                                                        allen_area_name,
                                                                                        subject_block.slice_number,
                                                                                        subject_block))

        return None

    # Get fUSI data
    original_fusi_times, original_fusi_data = subject_block.fusi_get_data()

    fusi_times, fusi_trace = detrend_fusi(original_fusi_times, original_fusi_data, mask=pmask,
                                          temporal_filter=temporal_filter, population_mean=True,
                                          temporal_filter_window=temporal_filter_window)
    fusi_trace = zscore(fusi_trace)

    # Get ephys data
    probe_roi_name = PROBE_ROI_NAMES[allen_area_name]
    ephys_probe = subject_block.ephys_get_probe_spikes_object(probe_name)
    probe_mask_min, probe_mask_max = \
        experiments.ephys_roi_markers[subject_block.meta_session.subject_name][
            subject_block.meta_session.session_name][probe_name][probe_roi_name]

    # Compute MUA
    nmua, mua_matrix = ephys_probe.time_locked_mua_matrix(
        fusi_times,
        dt=0.3,
        min_depthum=probe_mask_min,
        max_depthum=probe_mask_max,
        # if all clusters are bad, data hasn't been sorted -- so mark all clusters as usable.
        # good_clusters=np.ones_like(ephys_probe.good_clusters) if np.sum(ephys_probe.good_clusters)==0 else ephys_probe.good_clusters,
        chunk_sizeum=100)

    # Compute spikes
    fr = normalize_mua_matrix(mua_matrix)
    fr = zscore(fr)

    if hrf_convolve is not False:
        assert isinstance(hrf_convolve, np.ndarray)
        hrf_convolve = hrf_convolve.squeeze()
        hrfsize = len(hrf_convolve)
        fr = tikutils.hrf_convolution(fr, HRF=hrf_convolve).squeeze()

        # trim at beginning
        fr = zscore(fr[hrfsize:])
        fusi_times = fusi_times[hrfsize:]
        fusi_trace = zscore(fusi_trace[hrfsize:])

    return fusi_times, fusi_trace, fr
