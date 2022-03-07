'''
Handler data for audio-visual experiment.

20190727: Anwar Nunez-Elizalde
'''
import os
import time
from glob import glob
from functools import reduce


import datetime

import h5py
import numpy as np
import scipy.linalg
from scipy import interpolate
from scipy.stats import zscore


from tikreg import models, utils as tikutils
from fusi.extras import stats
from fusi.extras import readers

DATES = [20190730, 20190801]
HOURS = [1944, 1804]

DATASETS = {'CR015': [(20190730, 1944),
                      (20190801, 1804)]}


def date2cortexlab(date_tuple):
    dateob = datetime.datetime.strptime(str(date_tuple),
                                        '(%Y, %m, %d)')

    date = '%04i-%02i-%02i' % (dateob.year, dateob.month, dateob.day)
    return date


def datenumber2tuple(datenumber):
    '''20191231 -> (2019,12,13)
    '''
    assert isinstance(datenumber, int)
    dateob = datetime.datetime.strptime(str(datenumber),
                                        '%Y%m%d')
    time_tuple = dateob.timetuple()
    return time_tuple[:3]


def datetuple2date(date_tuple):
    dateob = datetime.datetime.strptime(str(date_tuple),
                                        '(%Y, %m, %d)')

    date = '%04i%02i%02i' % (dateob.year, dateob.month, dateob.day)
    return date


def dateidx2datetime(subject_name, dateidx=0):
    '''
    '''
    dataset = DATASETS[subject_name]
    date, hour = dataset[dateidx]
    dateob = datetime.datetime.strptime('%i %04i' % (date, hour),
                                        '%Y%m%d %H%M')
    return dateob


def hdf_load(hdf_file, key=None):
    '''Load a dataset from an HDF5 file

    Parameters
    ----------
    hdf_file : str
    key : str
        Dataset name

    Returns
    -------
    data_array : np.ndarray
        NumPy array containing HDF5 dataset
    '''
    assert os.path.exists(hdf_file)

    with h5py.File(hdf_file, 'r') as hfl:
        hdf_content = hfl.keys()
        if key is None:
            print('Please specify the HDF file content to load:', hdf_content)
            raise(ValueError('this'))
        assert key in hdf_content
        return np.asarray(hfl[key])


class BasicfUSi(object):
    def __init__(self, subject_name,
                 dateidx=0, session=0,
                 dataroot='/store/fast/subjects/CR015/'):
        '''
        '''
        dataset = DATASETS[subject_name]
        date, hour = dataset[dateidx]
        dateob = datetime.datetime.strptime('%i %04i' % (date, hour),
                                            '%Y%m%d %H%M')
        self.subject_name = subject_name
        self.dateob = dateob
        self.root = dataroot
        self.session = session
        self.dataset = dataset
        self.date = self.dateob.strftime('%Y%m%d')
        self.dateidx = dateidx
        self.load_fusi_dims()

    def __repr__(self):
        return '<fusi: %s %s>' % (self.subject_name, self.dateob.strftime('%Y%m%d'))

    def _get_fusi_fl(self):
        dateob = self.dateob
        year = '%04i' % dateob.year
        month = '%02i' % dateob.month
        day = '%02i' % dateob.day
        hours = '%02i%02i' % (dateob.hour, dateob.minute)

        pattern = '{yr}-{mo}-{day}_{hours}_CR015_YSStruct.mat'
        fusifl = pattern.format(yr=year, mo=month, day=day, hours=hours)
        fusifl = os.path.join(self.root, fusifl)
        return fusifl

    def get_fusi_object(self):
        fusifl = self._get_fusi_fl()
        fusi_object = MatlabHDF(fusifl)
        return fusi_object

    def get_fusi_shape(self):
        fusi_object = self.get_fusi_object()
        session = self.session
        fusi_data = fusi_object.load_content('fusi/dopplerFast', (session, 0))
        return fusi_data.shape[1:]

    def get_fusi_ticks(self):
        fusi_object = self.get_fusi_object()
        session = self.session
        xax = fusi_object.load_content('fusi/xAxis', (session, 0))[:, 0]
        zax = fusi_object.load_content('fusi/zAxis', (session, 0))[:, 0]
        yax = fusi_object.load_content('fusi/yCoord', (session, 0))[:, 0]
        return xax, yax, zax

    def get_fusi_dims(self):
        xax, yax, zax = self.get_fusi_ticks()
        xmm = np.mean(np.diff(xax))
        ymm = 0.0
        zmm = np.mean(np.diff(zax))
        return xmm, ymm, zmm

    def load_fusi_dims(self):
        self.dim_xmm, self.dim_ymm, self.dim_zmm = self.get_fusi_dims()
        self.dim_xax, self.dim_yax, self.dim_zax = self.get_fusi_ticks()
        # top = (self.dim_zax.max() - self.dim_zax.min())
        # bottom = (self.dim_xax.max() - self.dim_xax.min())
        self.image_aspect_ratio = self.dim_zmm/self.dim_xmm
        image_shape = self.get_fusi_shape()
        self.image_shape = image_shape

    def transform_fusi_image2twod(self, data):
        '''
        '''
        return data.reshape(data.shape[0], -1)

    def transform_fusi_twod2image(self, data):
        '''
        '''
        return data.reshape(data.shape[0],
                            self.image_shape[0],
                            self.image_shape[1])

    def show_fusi_image(self, oned, **kwargs):
        '''
        '''
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1)
        aspect = kwargs.get('aspect', self.image_aspect_ratio)
        kwargs['aspect'] = aspect
        im = ax.matshow(oned.reshape(self.image_shape).T, **kwargs)
        fig.suptitle('%s sess%02i' % (self.date, self.session), fontsize=20)
        return im

    def show_fusi_mosaic(self, twod, **kwargs):
        '''
        '''
        from matplotlib import pyplot as plt
        from fusi import viz
        aspect = kwargs.get('aspect', self.image_aspect_ratio)
        kwargs['aspect'] = aspect
        images = self.transform_fusi_twod2image(twod)
        o = viz.mosaic(images.T, -1, **kwargs)
        fig = plt.gcf()
        fig.suptitle('%s sess%02i' % (self.date, self.session), fontsize=20)
        return o

    def load_fusi_data(self):
        '''
        '''
        fusi_object = self.get_fusi_object()
        session = self.session
        print('Loading raw fUSi: %s sess%02i' % (self.date, self.session))

        fusi_data = fusi_object.load_content('fusi/dopplerFast', (session, 0))
        fusi_times = fusi_object.load_content('fusi/tAxisFast',
                                              (session, 0))[:].squeeze()
        fusi_image_shape = fusi_data.shape[1:]

        dur = np.diff(fusi_times).squeeze()
        stops = dur > 0.1
        acq_start = np.r_[fusi_times[0], fusi_times[1:][stops]]
        acq_end = np.r_[fusi_times[:-1][stops], fusi_times[-1]]
        assert len(acq_start) == len(acq_end)
        nfusi_bursts = acq_start.shape[0]

        self.fusi_raw = fusi_data
        self.fusi_times = fusi_times
        self.fusi_dt = np.mean(np.diff(fusi_times))
        self.fusi_image_shape = fusi_image_shape
        self.fusi_acq_start = acq_start
        self.fusi_acq_end = acq_end
        self.fusi_nbursts = nfusi_bursts


class BasicNeuroPixels(object):
    def __init__(self, subject_name,
                 dateidx=0, session=0,
                 dataroot='/store/fast/subjects/CR015/'):
        '''
        '''
        dataset = DATASETS[subject_name]
        date, hour = dataset[dateidx]
        dateob = datetime.datetime.strptime('%i %04i' % (date, hour),
                                            '%Y%m%d %H%M')
        self.subject_name = subject_name
        self.dateob = dateob
        self.root = dataroot
        self.session = session
        self.dataset = dataset
        self.date = self.dateob.strftime('%Y%m%d')
        self.dateidx = dateidx

    def __repr__(self):
        return '<neuropix: %s %s>' % (self.subject_name, self.dateob.strftime('%Y%m%d'))

    def _get_ephys_fl(self):
        date = self.dateob.strftime('%Y%m%d')
        session = self.session + 1  # 1-indexed
        ephysfl = '%s_eph_sess%i.mat' % (date, session)
        ephysfl = os.path.join(self.root, ephysfl)
        return ephysfl

    def load_ephys_data(self):
        ephysfl = self._get_ephys_fl()
        print('Loading ephys: %s sess%02i' % (self.date, self.session))
        matdat = matlab_data(ephysfl)['eph']
        spike_clusters = matdat.spikeCluster - 1  # change to 0-indexed

        self.spike_times = matdat.spikeTimes
        self.spike_clusters = spike_clusters
        self.spike_cluster_ids = np.arange(np.min(spike_clusters),
                                           np.max(spike_clusters)+1) - 1  # 0-indexed:
        self.spike_nclusters = len(self.spike_cluster_ids)
        self.spike_cluster_depths = matdat.clusterDepths

    def load_cluster_quality(self):
        '''
        '''
        date = self.dateob.strftime('%Y%m%d')
        cluster_group_fl = 'ephys_%s/kilosort/cluster_group.tsv' % date
        cluster_group_fl = os.path.join(self.root, cluster_group_fl)
        cluster_groups = np.loadtxt(cluster_group_fl, dtype='S')

        cluster_labels = np.asarray([t.decode()
                                    for t in cluster_groups[1:, 1]])
        cluster_idxs = np.asarray(
            cluster_groups[1:, 0], dtype=np.int)  # 0-indexed

        self.cluster_quality = {'labels': cluster_labels,
                                'idxs': cluster_idxs,
                                'ulabels': np.unique(cluster_labels)}


class RawSimultaneous(BasicfUSi, BasicNeuroPixels):
    def __init__(self, subject_name,
                 dateidx=0, session=0,
                 dataroot='/store/fast/subjects/CR015/'):
        '''
        '''
        super(RawSimultaneous, self).__init__(subject_name,
                                              dateidx=dateidx,
                                              session=session,
                                              dataroot=dataroot)


class SpatioTemporalMUA(BasicNeuroPixels):
    def __init__(self, subject_name,
                 dateidx=0, session=0,
                 dataroot='/store/fast/subjects/CR015/procdata',
                 probe_sizeum=3840):
        '''
        '''
        super(SpatioTemporalMUA, self).__init__(subject_name,
                                                dateidx=dateidx,
                                                session=session,
                                                dataroot=dataroot)

        self.probe_sizeum = probe_sizeum

    def __repr__(self):
        return '<MUA: %s %s>' % (self.subject_name, self.date)

    def _get_delay_spike_matrix_fl(self, delayms=0, binsizems=300):
        '''
        '''
        date = self.dateob.strftime('%Y%m%d')
        session = self.session

        pattern = 'spike_matrix_%s_sess%02i_binsize%04ims_%04ims.hdf'
        fl = pattern % (date, session, binsizems, delayms)
        return fl

    def load_delay(self, start_delayms=0, binsizems=300, recache=False):
        '''
        '''
        fl = self._get_delay_spike_matrix_fl(start_delayms, binsizems)
        fl = os.path.join(self.root, fl)
        if os.path.exists(fl) and recache is False:
            data = hdf_load(fl, 'data')
            times = hdf_load(fl, 'acq_start')  # seconds
            delay = hdf_load(fl, 'delay')     # seconds
            return data, times + delay

        print('Working on: %s...' % fl)
        subject = RawSimultaneous(self.subject_name,
                                  dateidx=self.dateidx,
                                  session=self.session)
        subject.load_ephys_data()
        subject.load_fusi_data()

        nclusters = subject.spike_nclusters
        nfusi_bursts = subject.fusi_nbursts
        acq_start = subject.fusi_acq_start
        acq_end = subject.fusi_acq_end
        spike_times = subject.spike_times
        spike_clusters = subject.spike_clusters
        delay = start_delayms/1000.0  # delay in seconds

        spike_matrix = np.zeros((nfusi_bursts, nclusters), dtype=np.int32)
        for burstidx, (astart, aend) in enumerate(zip(acq_start, acq_end)):
            # include first and last
            window_start = astart + delay
            if binsizems == 300:
                # "300" is short-hand exact to acquisition burst
                window_end = aend + delay
            else:
                # specify the window
                window_end = astart + delay + binsizems/1000.0

            spikes_valid = np.logical_and(spike_times >= window_start,
                                          spike_times < window_end)

            burst_spikes = spike_clusters[spikes_valid]
            for cdx, cluster in enumerate(np.unique(burst_spikes)):
                ncluster_spikes = (burst_spikes == cluster).sum()
                spike_matrix[burstidx, cluster] += ncluster_spikes
                if (cdx % 5 == 0) and (burstidx % 500 == 0):
                    print(self.subject_name, self.date, self.session,
                          delay, binsizems, window_start, window_end,
                          burstidx, cluster, ncluster_spikes)

        print('Saving: %s...' % fl)
        readers.hdf_dump(fl,
                         {'data': spike_matrix,
                          'delayms': start_delayms,
                          'delay': delay,
                          'binsizems': binsizems,
                          'acq_start': acq_start,
                          'acq_end': acq_end,
                          })
        return spike_matrix, acq_start + delay

    def load_cluster_depths(self):
        '''
        '''
        subject = RawSimultaneous(self.subject_name,
                                  dateidx=self.dateidx,
                                  session=self.session)
        subject.load_ephys_data()
        self.spike_cluster_depths = subject.spike_cluster_depths

    def load_mua(self,
                 chunk_size=200,
                 chunk_step=None,
                 start_delayms=0,
                 binsizems=300,
                 min_total_spikes=2000,
                 min_total_clusters=2,
                 clean_clusters=True,
                 dtype=np.float32,
                 recache=False):
        '''
        chunk_size : compute MUA across probe in chunks of size `chunk_size`
        '''
        spike_matrix, times = self.load_delay(start_delayms=start_delayms,
                                              binsizems=binsizems,
                                              recache=recache)

        min_spikes_criteria = spike_matrix.sum(0) > min_total_spikes

        if not hasattr(self, 'spike_cluster_depths'):
            self.load_cluster_depths()

        good_clusters = np.ones(spike_matrix.shape[-1]).astype(np.bool)
        if clean_clusters:
            subject = RawSimultaneous(self.subject_name,
                                      dateidx=self.dateidx,
                                      session=self.session)
            # only use clusters considered not noise
            subject.load_cluster_quality()
            cluster_labels = subject.cluster_quality['labels']
            cluster_idxs = subject.cluster_quality['idxs']

            # all are bad
            good_clusters = np.zeros(spike_matrix.shape[-1]).astype(np.bool)
            goods = np.logical_or(cluster_labels == 'mua',
                                  cluster_labels == 'good')
            good_idxs = cluster_idxs[goods]
            good_clusters[good_idxs] = True

        ##############################
        cluster_depths = self.spike_cluster_depths
        probe_size = self.probe_sizeum
        window = chunk_size  # um
        step = chunk_size if chunk_step is None else chunk_step  # um
        nchunks = int(np.ceil(probe_size/step))

        results = []
        depths = []
        nclusters = []
        for chunkidx in range(nchunks):
            min_depth = chunkidx*step
            # for last chunk, include clusters at 3840
            # achieved by increasing the probe size by a tiny amount
            max_depth = min(min_depth + window, probe_size + 1e-05)
            valid_clusters = np.logical_and(cluster_depths >= min_depth,
                                            cluster_depths < max_depth)

            valid_clusters = np.logical_and(valid_clusters, good_clusters)
            valid_clusters = np.logical_and(
                valid_clusters, min_spikes_criteria)

            # select clusters at this depth only
            mua_at_depth = spike_matrix[:, valid_clusters].sum(-1)

            if valid_clusters.sum() < min_total_clusters:
                # exclude mua at depth if the MUA only contains a few clusters
                mua_at_depth *= 0
                valid_clusters = np.zeros(len(valid_clusters), dtype=np.bool)

            results.append(mua_at_depth)
            # fix last chunk, max depth is delta bigger than actual probe size
            depths.append((min_depth, min(max_depth, probe_size)))
            nclusters.append(valid_clusters.sum())

        self.mua_matrix = np.asarray(results).T.astype(dtype)
        self.mua_depths = np.asarray(depths)
        self.mua_nclusters = np.asarray(nclusters)
        self.mua_times = times
        self.mua_mean = self.mua_matrix / self.mua_nclusters

    def show_mua(self, mua_data=None, xlims=(0, 1000), ylims=(0, 200), **figkwargs):
        from matplotlib import pyplot as plt
        if not hasattr(self, 'mua_matrix'):
            self.load_mua()

        mua_matrix = self.mua_matrix if mua_data is None else mua_data
        mua_depths = self.mua_depths
        mua_nclusters = self.mua_nclusters

        fig, axes = plt.subplots(
            mua_matrix.shape[1], 1, sharey=True, sharex=True, **figkwargs)
        times = self.mua_times

        for depthidx in range(mua_matrix.shape[1]):
            ax = axes[depthidx]
            ax.plot(mua_matrix[:, depthidx], color='C%i' % depthidx,
                    label='#%i: %04i-%04ium' % tuple([depthidx]+list(mua_depths[depthidx])))
            ax.set_ylabel('MUA\n(n=%i)' % mua_nclusters[depthidx])
            ax.legend(loc='upper right')

        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)
        ax.set_xlabel('time [sec]')
        fig.suptitle('%s sess%02i' % (self.date, self.session), fontsize=20)
        plt.tight_layout()
        return ax


class ProcfUSi(BasicfUSi):
    def __init__(self, subject_name,
                 dateidx=0, session=0,
                 dataroot='/store/fast/subjects/CR015/',
                 dataout='/store/fast/subjects/CR015/procdata',
                 ):
        '''
        '''
        super(ProcfUSi, self).__init__(subject_name,
                                       dateidx=dateidx,
                                       session=session,
                                       dataroot=dataroot)
        self.dataout = dataout
        self.dataroot = dataroot

    def _get_fusi_burst_fl(self):
        fl = 'fusi_burst_matrix_%s_sess%02i.hdf' % (self.date, self.session)
        fl = os.path.join(self.dataout, fl)
        return fl

    def _get_fusi_filtered_fl(self, filter_name='median', window=9):
        pattern = 'fusi_burst_matrix_%sFilter_window%i_%s_sess%02i.hdf'
        fl = pattern % (filter_name, window, self.date, self.session)
        fl = os.path.join(self.dataout, fl)
        return fl

    def _get_fusi_goodframes_fl(self):
        pattern = 'fusi_burst_matrix_goodframes_%s_sess%02i.hdf'
        fl = pattern % (self.date, self.session)
        fl = os.path.join(self.dataout, fl)
        return fl

    def _get_fusi_smooth_fl(self, sigma_um=50):
        pattern = 'fusi_burst_matrix_smooth_sigma_%04ium_%s_sess%02i.hdf'
        fl = pattern % (sigma_um, self.date, self.session)
        fl = os.path.join(self.dataout, fl)
        return fl

    def load_fusi_smooth_data(self, sigma_um=20,
                              dtype=np.float32,
                              recache=True,
                              **gframeskwargs):
        '''
        '''
        from scipy import ndimage

        fl = self._get_fusi_smooth_fl(sigma_um=sigma_um)

        if os.path.exists(fl) and recache is False:
            print('Overwriting data with %i[um] smooth version' % sigma_um)
            self.fusi_data = hdf_load(fl, 'data')
            self.fusi_times = hdf_load(fl, 'acq_start')  # seconds
            self.fusi_dt = np.mean(np.diff(self.fusi_times))
            self.fusi_smooth_um = hdf_load(fl, 'sigma_um')
            return

        self.load_fusi_goodframes_data(recache=recache, **gframeskwargs)
        image_data = self.transform_fusi_twod2image(
            self.fusi_data).astype(dtype)
        smooth_data = np.zeros_like(image_data)
        sigma_pix = sigma_um/(self.dim_xmm*1000)
        sigmas = (sigma_pix, sigma_pix/self.image_aspect_ratio)
        print('Smoothing:', sigma_um, sigmas)

        for imageidx in range(image_data.shape[0]):
            if imageidx % 100 == 0:
                print(imageidx)
            # BF: INDENT! 20190830 2053
            inim = image_data[imageidx]
            outim = ndimage.gaussian_filter(inim, sigma=sigmas)
            smooth_data[imageidx] = outim
        smooth_data = self.transform_fusi_image2twod(smooth_data)

        print('Saving: %s...' % fl)
        readers.hdf_dump(fl,
                         {'data': smooth_data,
                          'acq_start': self.fusi_times,
                          'sigma_um': sigma_um,
                          'sigmas': sigmas})
        print('Overwriting data with %i[um] smooth version' % sigma_um)
        print(sigma_um, sigmas)
        self.fusi_data = smooth_data

    def load_fusi_burst_data(self, trim=(1, -1), recache=False):
        '''
        trim : tuple
            Drops the N first and M last samples from the burst
        '''
        fl = self._get_fusi_burst_fl()

        if os.path.exists(fl) and recache is False:
            data = hdf_load(fl, 'data')
            times = hdf_load(fl, 'acq_start')  # seconds
            return data, times

        print('Working on: %s...' % fl)
        subject = RawSimultaneous(self.subject_name,
                                  dateidx=self.dateidx,
                                  session=self.session)
        subject.load_fusi_data()

        fusi_data = subject.fusi_raw[:]  # read data
        fusi_times = subject.fusi_times
        acq_start = subject.fusi_acq_start
        acq_end = subject.fusi_acq_end
        fusi_image_shape = subject.fusi_image_shape
        nfusi_bursts = subject.fusi_nbursts
        trim_beg, trim_end = trim
        burst_nframes = []

        fusi_bursts_mean = np.zeros((nfusi_bursts,
                                     fusi_image_shape[0],
                                     fusi_image_shape[1]), dtype=np.float32)
        fusi_bursts_snr = np.zeros((nfusi_bursts,
                                    fusi_image_shape[0],
                                    fusi_image_shape[1]), dtype=np.float32)

        for burstidx, (astart, aend) in enumerate(zip(acq_start, acq_end)):
            fusi_valid = np.logical_and(fusi_times >= astart,
                                        fusi_times <= aend)
            burst_nframes.append(fusi_valid.sum())
            # drop beg/end frames
            burst_mean = np.mean(fusi_data[fusi_valid][trim_beg:trim_end], 0)
            burst_std = np.std(fusi_data[fusi_valid][trim_beg:trim_end], 0)

            fusi_bursts_mean[burstidx] = burst_mean.astype(np.float32)
            fusi_bursts_snr[burstidx] = burst_mean/burst_std.astype(np.float32)

            if burstidx % 250 == 0:
                print(self.subject_name, self.date, self.session,
                      burstidx, astart, aend, fusi_valid.sum())

        print('Saving: %s...' % fl)
        readers.hdf_dump(fl,
                         {'data': fusi_bursts_mean,
                          'snr': fusi_bursts_snr,
                          'burst_nframes': np.asarray(burst_nframes),
                          'acq_start': acq_start,
                          'acq_end': acq_end,
                          'fusi_times': fusi_times,
                          'fusi_image_shape': fusi_image_shape,
                          'nfusi_bursts': nfusi_bursts})
        return fusi_bursts_mean, acq_start

    def load_fusi_burst_snr(self):
        fl = self._get_fusi_burst_fl()
        fl = os.path.join(self.root, fl)
        snr = hdf_load(fl, 'snr')
        times = hdf_load(fl, 'acq_start')
        return snr, times

    def show_fusi_snr(self, snr_threshold=50, snr_std=2, ax=None):
        from matplotlib import pyplot as plt
        fusi_snr, fusi_times = self.load_fusi_burst_snr()
        fusi_snr = fusi_snr.reshape(fusi_snr.shape[0], -1)

        snr_baseline = np.percentile(fusi_snr, snr_threshold, axis=1)
        snr_minimum = snr_baseline.mean() - snr_baseline.std()*snr_std
        good_frames = snr_baseline > snr_minimum
        print('Found #%i bad frames' % np.logical_not(good_frames).sum())

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.plot(fusi_times, snr_baseline)
        ax.vlines(fusi_times[np.logical_not(good_frames)],
                  snr_baseline.min(), snr_baseline.mean())
        ax.hlines(snr_minimum, fusi_times.min(), fusi_times.max(), color='red')
        return ax

    def load_fusi_goodframes_data(self,
                                  snr_threshold=50, snr_std=2,
                                  dtype=np.float32,
                                  recache=False):
        '''
        '''
        fl = self._get_fusi_goodframes_fl()
        fl = os.path.join(self.root, fl)

        if os.path.exists(fl) and recache is False:
            good_frames = hdf_load(fl, 'good_frames')
            self.fusi_good_frames = good_frames
            self.fusi_data = hdf_load(fl, 'data')
            self.fusi_times = hdf_load(fl, 'acq_start')  # seconds
            self.fusi_dt = np.mean(np.diff(self.fusi_times))
            self.fusi_snr_global = hdf_load(fl, 'snr_baseline')
            self.fusi_snr_minimum = hdf_load(fl, 'snr_minimum')

            nframes = self.fusi_data.shape[0]
            nbad_frames = nframes - len(good_frames)
            pct_bad_frames = nbad_frames/nframes
            info = (nbad_frames, nframes, pct_bad_frames*100)
            print('Dropped %i/%i bad frames (%0.03fpct)' % info)
            return

        print('Working on: %s...' % fl)
        fusi_data, fusi_times = self.load_fusi_burst_data()
        fusi_snr, _ = self.load_fusi_burst_snr()
        # flatten
        fusi_image_shape = fusi_data.shape[1:]
        fusi_data = fusi_data.reshape(fusi_data.shape[0], -1)
        fusi_snr = fusi_snr.reshape(fusi_snr.shape[0], -1)

        snr_baseline = np.percentile(fusi_snr, snr_threshold, axis=1)
        snr_minimum = snr_baseline.mean() - snr_baseline.std()*snr_std
        good_frames = snr_baseline > snr_minimum
        print('Found #%i bad frames' % np.logical_not(good_frames).sum())

        newdat = np.zeros_like(fusi_data)
        for chidx in range(fusi_data.shape[-1]):
            if chidx % 1000 == 0:
                print('Resampling pixel #%i' % chidx)
            spline = interpolate.splrep(fusi_times[good_frames],
                                        fusi_data[good_frames, chidx])
            dd = interpolate.splev(fusi_times, spline)
            newdat[:, chidx] = dd

        print('Saving %s...' % fl)
        readers.hdf_dump(fl,
                         {'data': newdat,
                          'acq_start': fusi_times,
                          'good_frames': good_frames.nonzero()[0],
                          'snr_baseline': snr_baseline,
                          'snr_minimum': snr_minimum,
                          'snr_threshold': snr_threshold,
                          'snr_std': snr_std})

        self.fusi_data = newdat.astype(dtype)
        self.fusi_times = fusi_times
        self.fusi_dt = np.mean(np.diff(fusi_times))
        self.fusi_good_frames = good_frames.nonzero()[0]
        self.fusi_snr_global = snr_baseline
        self.fusi_snr_minimum = snr_minimum

    def load_nonbrain_mask(self):
        '''
        '''
        details = self.date, self.dim_yax
        maskfl = os.path.join(
            self.dataroot, '%s_%0.2fmm_outbrainmask.hdf' % details)
        print(maskfl)
        mask = readers.hdf_load(maskfl, 'mask').astype(np.bool)
        return mask

    def remove_nonbrain_nuisance(self,
                                 npcs=10,
                                 R2thresh=-np.inf,
                                 pctsignalchange=False,
                                 **good_frames_kwargs):
        '''
        '''
        self.load_fusi_goodframes_data(**good_frames_kwargs)
        fusi_data = self.fusi_data.copy()
        if pctsignalchange:
            from fusi.preproc import preproc_loader
            fusi_data = preproc_loader.pixelwise_pctsignalchange(fusi_data)

        mask = self.load_nonbrain_mask().ravel()
        # only keep temporal PCs
        U = scipy.linalg.svd(fusi_data[:, mask], full_matrices=False)[0]
        U = U[:, :npcs]
        Yhat = models.olspred(U, fusi_data)
        R2 = tikutils.columnwise_rsquared(Yhat, fusi_data)
        bad_pixels = R2 > R2thresh
        fusi_data[:, bad_pixels] = fusi_data[:,
                                             bad_pixels] - Yhat[:, bad_pixels]
        return fusi_data


class SimultaneousfUSiNeuropixels(object):
    def __init__(self, subject_name,
                 dateidx=0, session=0, autoload=True):
        dataset = DATASETS[subject_name]
        date, hour = dataset[dateidx]
        dateob = datetime.datetime.strptime('%i %04i' % (date, hour),
                                            '%Y%m%d %H%M')

        self.subject_name = subject_name
        self.session = session
        self.dateidx = dateidx

        self.dateob = dateob
        self.dataset = dataset
        self.date = self.dateob.strftime('%Y%m%d')
        self.load_properties()

        if autoload is True:
            self.load()

    def __repr__(self):
        info = (self.subject_name, self.date, self.session)
        return '<simultaneous fUSi/NP: subject=%s date=%s session=%s>' % info

    def load_probe_roi_mask(self):
        '''
        '''
        details = self.dateob.strftime(
            '%Y-%m-%d_%H%M'), '%0.0fNPmask' % (self.fusi_dim_yax*10)
        maskfl = os.path.join(self.fusi.dataout, '%s_%s.hdf' % (details))
        print(maskfl)
        mask = readers.hdf_load(maskfl, 'mask').astype(np.bool)
        probe_mask = mask.copy()
        return probe_mask

    def load_eye_video_buffer(self):
        '''
        '''
        pattern = '/cortexlab/zserver/Data/Subjects/{subject}/{year}-{month}-{day}/{session_oneidx}'
        root = pattern.format(subject=self.subject_name,
                              year=self.dateob.year,
                              month='%02i' % self.dateob.month,
                              day='%02i' % self.dateob.day,
                              session_oneidx=self.session+1)
        # '2019-08-01_4_CR015_eye.mj2'
        flpattern = '{year}-{month}-{day}_{session_oneidx}_{subject}_eye.mj2'
        fl = flpattern.format(subject=self.subject_name,
                              year=self.dateob.year,
                              month='%02i' % self.dateob.month,
                              day='%02i' % self.dateob.day,
                              session_oneidx=self.session+1)
        flname = os.path.join(root, fl)
        assert os.path.exists(flname)
        from moten.io import video_buffer
        return video_buffer(flname)

    def load_eyetracking(self):
        '''
        '''
        pattern = '/cortexlab/zserver/Data/Subjects/{subject}/{year}-{month}-{day}/{session_oneidx}'
        root = pattern.format(subject=self.subject_name,
                              year=self.dateob.year,
                              month='%02i' % self.dateob.month,
                              day='%02i' % self.dateob.day,
                              session_oneidx=self.session+1)
        # '2019-08-01_4_CR015_eye.mj2'
        flpattern = '{year}-{month}-{day}_{session_oneidx}_{subject}_eye_processed.mat'
        fl = flpattern.format(subject=self.subject_name,
                              year=self.dateob.year,
                              month='%02i' % self.dateob.month,
                              day='%02i' % self.dateob.day,
                              session_oneidx=self.session+1)
        flname = os.path.join(root, fl)
        assert os.path.exists(flname)
        data = matlab_data(flname)['results']
        area = data.area.copy()
        xdim = data.x.copy()
        ydim = data.y.copy()
        fusi_object = self.fusi.get_fusi_object()
        times = np.asarray(fusi_object.load_content(
            'fusi/eyeTimes', (self.session, 0))).squeeze()
        return times, xdim, ydim, area

    def load(self, fusikwargs={}, muakwargs={}):
        '''
        '''
        subject_name = self.subject_name
        dateidx = self.dateidx
        session = self.session

        # fusi
        self.fusi = ProcfUSi(subject_name,
                             dateidx=dateidx,
                             session=session)

        self.fusi.load_fusi_goodframes_data(**fusikwargs)

        # MUA
        self.neuropixels = SpatioTemporalMUA(subject_name,
                                             dateidx=dateidx,
                                             session=session)
        self.neuropixels.load_mua(**muakwargs)

    def load_smooth(self, sigma_um=100):
        self.fusi.load_fusi_smooth_data(sigma_um)

    def load_properties(self):
        subject_name = self.subject_name
        dateidx = self.dateidx
        session = self.session
        # image shape information
        raw = RawSimultaneous(subject_name,
                              dateidx=dateidx,
                              session=session)

        self.fusi_dim_xmm, self.fusi_dim_ymm, self.fusi_dim_zmm = raw.get_fusi_dims()
        self.fusi_dim_xax, self.fusi_dim_yax, self.fusi_dim_zax = raw.get_fusi_ticks()
        top = (self.fusi_dim_zax.max() - self.fusi_dim_zax.min())
        bottom = (self.fusi_dim_xax.max() - self.fusi_dim_xax.min())
        self.fusi_aspect_ratio = top/bottom

        image_shape = raw.get_fusi_shape()
        self.fusi_image_shape = image_shape


# subraw = RawSimultaneous('CR015')
# submua = SpatioTemporalMUA('CR015')
# sub = ProcfUSi('CR015')

def vec2image(dat, imshape):
    return dat.reshape(imshape[0], imshape[1])


def arr2image(dat, imshape):
    return dat.reshape(dat.shape[0], imshape[0], imshape[1])


def lanczosfun(cutoff, t, window=3):
    """Compute the lanczos function with some cutoff frequency [B]Hz at some time [t].
    [t] can be a scalar or any shaped numpy array.
    If given a [window], only the lowest-order [window] lobes of the sinc function
    will be non-zero.
    """
    t = np.atleast_1d(t * cutoff)
    val = window * np.sin(np.pi*t) * np.sin(np.pi*t/window) / (np.pi**2 * t**2)
    val[t == 0] = 1.0
    val[np.abs(t) > window] = 0.0
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
    print("Doing lanczos interpolation with cutoff=%0.3f and %d lobes." %
          (cutoff, window))

    ## Build up sinc matrix ##
    sincmat = np.zeros((len(newtime), len(oldtime)), dtype=np.float32)
    for ndi in range(len(newtime)):
        sincmat[ndi, :] = lanczosfun(cutoff, newtime[ndi]-oldtime, window)

    if rectify:
        newdata = np.hstack([np.dot(sincmat, np.clip(data, -np.inf, 0)),
                             np.dot(sincmat, np.clip(data, 0, np.inf))])
    else:
        ## Construct new signal by multiplying the sinc matrix by the data ##
        newdata = np.dot(sincmat, data)

    return newdata


def lanczosinterp1D(data, oldtime, newtime, window=3, cutoff=1.0, rectify=False):
    """lite mem
    """
    print("Doing lanczos interpolation with cutoff=%0.3f and %d lobes." %
          (cutoff, window))

    ## Build up sinc matrix ##
    newdata = np.zeros((len(newtime), 1), dtype=np.float32)
    for ndi in range(len(newtime)):
        for oldi in range(len(oldtime)):
            newdata[ndi] = lanczosfun(cutoff,
                                      newtime[ndi]-oldtime[oldi],
                                      window)*data[oldi]

    return newdata


def matlab_data(matfile):
    '''
    '''
    from scipy import io as sio
    try:
        matdat = sio.loadmat(matfile,
                             struct_as_record=False,
                             verify_compressed_data_integrity=False,
                             squeeze_me=True)
    except OSError:
        nsec = np.random.rand()*5
        time.sleep(nsec)
        matdat = sio.loadmat(matfile,
                             struct_as_record=False,
                             verify_compressed_data_integrity=False,
                             squeeze_me=True)

    return matdat


def stimulus_vector(matdat, event, location=None, correct_only=False):
    '''
    '''
    def get_value(key, location=location, left=-60, right=60):
        '''
        '''
        assert location is not None
        value = getattr(matdat['blk'], key)
        matcher = left if location == 'L' else right
        value = value == matcher
        return value

    if event == 'visual':
        times = getattr(matdat['fus'], 'visStimPeriodOnOff')
        value = get_value('visInitialAzimuth', location=location)

    elif event == 'auditory':
        times = getattr(matdat['fus'], 'audStimPeriodOnOff')
        value = get_value('audInitialAzimuth', location=location)

    elif event == 'motor':
        times = getattr(matdat['fus'], 'movementTimes')
        value = get_value('responseMade', location=location,
                          left=1, right=2)

    elif event == 'reward':
        times = getattr(matdat['fus'], 'rewardTimes')
        values = getattr(matdat['blk'], 'feedback')

        if location == 'correct':
            value = values == 1
        elif location == 'incorrect':
            value = values == -1
        elif location == 'timeout':
            value = values == 0
        else:
            raise ValueError('Unknown: %s' % location)

        if times.dtype.str == '|O':
            # only the first time.
            # multiple corresponds to experimenter-triggered reward
            # if there are multiple, this is likely a time-out
            print('Multiple rewards')
            ntimes = []
            for time in times:
                if np.isscalar(time):
                    ntimes.append(time)
                elif len(time) == 0:
                    ntimes.append(np.nan)
                elif len(time) > 0:
                    ntimes.append(time[0])
                else:
                    ValueError(time)
                    times = np.asarray(ntimes)

    if correct_only:
        valid_trials = getattr(matdat['blk'], 'feedback') == 1
        print('Using only valid trials: n=%i' % valid_trials.sum())
        assert valid_trials.shape[0] == times.shape[0]
        times = times[valid_trials]
        value = value[valid_trials]

    return times, value.astype(np.float)


class MatlabHDF(object):
    def __init__(self, filename, verbose=False):
        '''
        '''
        self.verbose = verbose
        self.filename = filename
        self.hdf5 = h5py.File(self.filename, 'r')
        clist = list(self.hdf5.keys())
        if self.verbose:
            print('%s contains (%i items):' % (filename, len(clist)))
            print(clist)
        try:
            self.nsess = self.hdf5['fusi/doppler'].shape[0]
        except:
            self.nsess = self.hdf5['doppler'].shape[0]
        for idx in range(self.nsess):
            if verbose:
                print('fusi: sess=%i' % idx)
            try:
                self.session_info((idx, 0))
            except ValueError:
                if verbose:
                    print('No "fusi/block" information (MPEP)')

    def __repr__(self):
        desc = self.hdf5['ExpRef'][:].squeeze()
        desc = ''.join([chr(t) for t in desc])
        return '<%s: %s>' % (self.filename, desc)

    def session_info(self, session=(0, 0)):
        '''
        '''
        try:
            ob = self.load_content('fusi/block', session, verbose=self.verbose)
        except:
            ob = self.load_content('block', session, verbose=self.verbose)
        if self.verbose:
            print(''.join([chr(t) for t in ob['expDef'][:].squeeze()]))
        mmpos = float(self.load_content('fusi/yCoord', session)[:])
        if self.verbose:
            print('Coordinate: %0.04f[mm]' % mmpos)

    def load_content(self, path, index=None, verbose=False):
        '''
        path:
            path/to/the/thing
        '''
        hdf_content = self.hdf5[path]

        # read object
        dtype = hdf_content.dtype
        if dtype.str == '|O':
            if index is None:
                print(path, hdf_content.shape, hdf_content)
                txt = 'Specify index of reference object to load'
                print(txt)
                return hdf_content
            else:
                if verbose:
                    print(path, hdf_content.shape, hdf_content.dtype)
                ref = hdf_content[index]
                hdf_content = self.hdf5[ref]
                if verbose:
                    print('Loading:', index)
                if isinstance(hdf_content, h5py.Dataset):
                    if verbose:
                        print(hdf_content.shape, hdf_content.dtype)
                if isinstance(hdf_content, h5py.Group):
                    if verbose:
                        print('\n'.join(list(hdf_content.keys())))
        return hdf_content

    def ls(self, *args):
        path = args[0]
        rest = args[1:]
        print(path, rest)
        hdf_content = self.hdf5[path]

        if isinstance(hdf_content, h5py.Group):
            print('"%s" contains:' % path)
            print('\n'.join(list(hdf_content.keys())))
        elif isinstance(hdf_content, h5py.Dataset):
            print('"%s" is a dataset' % path, hdf_content.shape)


def get_sessions(subject, year, month, day, root='/cortexlab/zubjects/Subjects/', matcher='*'):
    date = '{}-{}-{}'.format(year, month, day)
    path = os.path.join(root, subject, date)
    sessions = glob(os.path.join(path, matcher))
    return sessions


class MultiWorldFUSI(object):
    def __init__(self, subject_name, date=None, sessidx=None, verbose=False):
        '''
        '''
        fusi_pattern = '/cortexlab/zserver/Lab/Share/Anwar/fUSiData/{subject}/*YSStruct.mat'
        fusi_files = glob(fusi_pattern.format(subject=subject_name))
        fusi_dates = [tuple(path.split('/')[-1].split('.')
                            [0].split('_')[0].split('-')) for path in fusi_files]
        date_codes = ['%s%s%s' % date for date in fusi_dates]
        date2fusi = {k: v for k, v in zip(date_codes, fusi_files)}

        date_sessions = {}

        behavior_pattern = '/cortexlab/zubjects/Subjects/{subject}/{year}-{month}-{day}/*/*ProcBlock.mat'
        date2behavior = {}
        for date in fusi_dates:
            datecode = '%s%s%s' % date
            year, month, day = date
            behavior_files = glob(behavior_pattern.format(subject=subject_name,
                                                          year=year,
                                                          month=month,
                                                          day=day))
            date2behavior[datecode] = sorted(behavior_files)
            day_sessions = {sess.split('/')[-1]: i for i, sess in enumerate(get_sessions(subject_name, year, month, day))
                            if len(sess.split('/')[-1]) < 2}
            valid_sessions = [fl.split('_')[1] for fl in behavior_files]
            date_sessions[datecode] = {
                sess: day_sessions[sess] for sess in valid_sessions}

        if verbose:
            for date in sorted(list(date2fusi.keys())):
                print(date)
                print(date2fusi[date])
                print('\n'.join(date2behavior[date]))

        self._date_sessions = date_sessions
        # self._sessions = sessions
        self._date2behavior = date2behavior
        self._date2fusi = date2fusi

        ##
        self.subject_name = subject_name
        self.fusi_session = None
        self.fusi_date = None

    def view_session_names(self, DATE):
        '''Print the name of the experiments recorded during date.
        '''

        for sessidx in range(len(self._date_sessions[DATE])):
            sessions = self._date_sessions[DATE]
            SESSION = list(sessions.values())[sessidx]

            index2sess = {v: k for k, v in sessions.items()}
            sessnum = index2sess[SESSION]
            matfile = self._date2behavior[DATE][sessidx]

            matdat = matlab_data(matfile)
            info = (matfile, self.subject_name, DATE,
                    sessidx, getattr(matdat['blk'], 'expDef'))
            print('[%s] - %s on %s (sessidx=%i): %s' % info)

    def get_session_names(self, DATE):
        '''Extract the experiment name and its index into the fusi data
        '''
        out = {}
        for sessidx in range(len(self._date_sessions[DATE])):
            sessions = self._date_sessions[DATE]
            SESSION = list(sessions.values())[sessidx]

            index2sess = {v: k for k, v in sessions.items()}
            sessnum = index2sess[SESSION]
            matfile = self._date2behavior[DATE][sessidx]
            matdat = matlab_data(matfile)
            out[getattr(matdat['blk'], 'expDef')] = sessidx
        return out

    def get_experiment_dates(self):
        '''Extract the dates of all experiments available for fusi data
        '''
        return sorted(list(self._date2fusi.keys()))

    def view_experiment_dates(self):
        '''Print all dates of fusi experiemtns
        '''
        print('Available experiments:')
        print(', '.join(self._date2fusi.keys()))

    def load_fusi_date(self, date):
        '''Load MATLAB HDF file containing fusi data from that date
        '''

        assert date in self._date2fusi
        flname = self._date2fusi[date]
        self.fusi_object = MatlabHDF(flname)
        shape = dataobj.load_content('fusi/dopplerFast', (0, 0))

        self.fusi_image_shape = shape[1:]
        self.fusi_session = None
        self.fusi_date = date

    def get_session_fusi(self, date, session, normalize=True):
        '''Load time and data array for an experiment
        '''
        self.load_fusi_date(date)
        ob = self.fusi_object.load_content('fusi/dopplerFast', (session, 0))
        fusi_signals = ob[:]    # read data object
        if normalize:
            # normalize to range: [0-100]
            fusi_signals -= fusi_signals.min()
            fusi_signals /= fusi_signals.max()
            fusi_signals *= 100

        fusi_times = dataobj.load_content(
            'fusi/tAxisFast', (session, 0))[:].squeeze()
        return fusi_times, fusi_signals

    def load_session_behavior(self, DATE, session):
        '''Load MATLAB MAT file containing experiment details for date-session
        '''
        # index in fusi data HDF vs. the folder name on server
        sessions = self._date_sessions[DATE]
        absolute_session_index = list(sessions.values())[session]

        index2sess = {v: k for k, v in sessions.items()}
        sessnum = index2sess[absolute_session_index]
        matfile = self._date2behavior[DATE][session]
        matdat = matlab_data(matfile)
        print(matfile, getattr(matdat['blk'], 'expDef'), session)
        self.matdat = matdat
        self.blk = matdat['blk']
        self.fus = matdat['fus']

    def get_session_behavior(self, date, session, **kwargs):
        '''Extract experimental features for each trial in the date-session

        kwargs:
            correct_only : only return correct trials
        '''

        self.meta_features = [('visual', 'L'), ('visual', 'R'),
                              ('auditory', 'L'), ('auditory', 'R'),
                              ('motor', 'L'), ('motor', 'R'),
                              ('reward', 'correct')]

        self.load_session_behavior(date, session)
        matdat = self.matdat

        trials = {}
        for edx, (event, location) in enumerate(self.meta_features):
            times, feature_vec = stimulus_vector(
                matdat, event, location, **kwargs)
            trials['%s_%s' % (event, location)] = (times, feature_vec)
        return trials
