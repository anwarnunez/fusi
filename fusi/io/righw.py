import os
import pathlib
import itertools

import numpy as np
from fusi.config import DATA_ROOT

from fusi.utils import band_pass_signal


def can_int(val):
    try:
        fval = float(val)
        return fval == int(val)
    except:
        return False


def datetuple2isoformat(date_tuple):
    '''
    '''
    year, month, day = date_tuple[:3]
    return '{year}-{month}-{day}'.format(year=year,
                                         month='%02i' % month,
                                         day='%02i' % day)


def isoformat_filename2info(flname):
    '''Extract subject, date and block number from file name

    E.g. 2020-11-01_13_CR024_data.csv -> ('CR024', '2020-11-01', '13')

    Parameters
    ----------
    flname : str or pathlib.path

    Returns
    -------
    subject : str
    date : str
    blocknum : str
    '''
    stem = pathlib.Path(flname).stem
    date, blocknum, subject = stem.split('_')[:3]
    return subject, date, blocknum


def isoformat_filename2fullpath(flname,
                                root=DATA_ROOT):
    '''Convert a cortex-lab formated filename to a full path

    E.g. 2020-11-01_13_CR024_data.csv gets translated to:
    /root/subject/date/block_number/2020-11-01_13_CR024_data.csv

    Parameters
    ----------
    flname : str or pathlib.Path

    Returns
    -------
    outpath : pathlib.Path

    Examples
    --------
    >>> isoformat_filename2fullpath('2020-11-01_13_CR024_data.csv')
    PosixPath('/DATA_ROOT/CR024/2020-11-01/13/2020-11-01_13_CR024_data.csv')
    '''
    stem = pathlib.Path(flname).stem
    date, blocknum, subject = stem.split('_')[:3]

    outpath = pathlib.Path(root).joinpath(subject, date, blocknum)
    return outpath / pathlib.Path(flname).name


class DataStore(object):
    def __init__(self, subject='CR01', expnum=1, date=(2017, 11, 17), root=DATA_ROOT):
        self.subject = subject
        self.date = date
        self.root = root
        self._set_file_paths(expnum)
        self.isoformat_date = datetuple2isoformat(date)

        year, month, day = date
        path_pattern = '{root}/{subject}/{year}-{month}-{day}/{expnum}/'
        path = path_pattern.format(root=root,
                                   year=year,
                                   month='%02i' % month,
                                   day='%02i' % day,
                                   expnum=expnum,
                                   subject=subject)
        self.__path__ = path
        self.__pathobj__ = pathlib.Path(path)

    def __repr__(self):
        details = (self.subject, self.date, self.expnum)
        year, month, day = self.date
        description = '<experiment dataset: experiment #{sl}. {sub} ({yr}/{mo}/{day})>'
        return description.format(sub=self.subject,
                                  sl=self.expnum,
                                  yr=year, mo=month, day=day)

    def _set_file_paths(self, expnum):
        '''
        '''
        self.expnum = expnum

        # file template
        template = '{root}/{subject}/{year}-{month}-{day}/{expnum}/{year}-{month}-{day}_{expnum}_{subject}_{dtype}.mat'
        dtypes = ['Timeline']

        year, month, day = self.date
        files = {dt: template.format(root=self.root,
                                     year=year,
                                     month='%02i' % month,
                                     day='%02i' % day,
                                     subject=self.subject,
                                     expnum=self.expnum,
                                     dtype=dt)
                 for dt in dtypes}

        # protocol_pattern = '{root}/{subject}/{year}-{month}-{day}/{expnum}/Protocol.mat'
        # files['Protocol']  = protocol_pattern.format(root=self.root,
        #                                              year=year,
        #                                              month='%02i'%month,
        #                                              day='%02i'%day,
        #                                              expnum=self.expnum,
        #                                              subject=self.subject)

        # store paths
        for fll in files.values():
            assert os.path.exists(fll)
        self.files = files
        self.dtypes = list(files.keys())

    @property
    def available_sessions(self):
        '''
        '''
        return os.listdir(os.path.join(self.root, self.subject))

    @property
    def available_experiments(self):
        '''
        '''
        items = os.listdir(os.path.join(self.root,
                                        self.subject,
                                        self.isoformat_date))
        # str_items = sorted([item for item in items if not can_int(item)])
        int_items = sorted(
            [int(item) for item in items if can_int(item) and int(item) < 1000])
        return int_items  # + str_items

    def set_experiment(self, expnum):
        '''
        '''
        self._set_file_paths(expnum)

    def load_data(self, dtype):
        '''
        '''
        from scipy import io as sio
        assert dtype in self.dtypes
        dtype2keys = {'fus': 'doppler',
                      'Protocol': 'Protocol',
                      'Timeline': 'Timeline',
                      }

        if dtype not in dtype2keys:
            raise IOError("Give info on how to load '%s':\n%s" %
                          (dtype, self.files[dtype]))

        key = dtype2keys[dtype]
        try:
            dat = sio.loadmat(self.files[dtype], struct_as_record=False,
                              verify_compressed_data_integrity=False)[key][0, 0]
        except OSError:
            # Assume I/O latency issues and try again
            import time
            nsec = np.random.rand()*5
            time.sleep(nsec)
            dat = sio.loadmat(self.files[dtype], struct_as_record=False,
                              verify_compressed_data_integrity=False)[key][0, 0]
        return dat


class ExperimentData(DataStore):
    def __init__(self, *args, **kwargs):
        super(ExperimentData, self).__init__(*args, **kwargs)
        self.load_timeline()
        # self.load_photodiode_events()
        # self.load_stimulus_events()
        # self.load_protocol()

    def load_timeline(self):
        '''
        '''
        timeline = self.load_data('Timeline')
        HWEVENTS = [t.name[0] for t in timeline.hw[0, 0].inputs[0]]
        self.hwevents = HWEVENTS
        self.timeline = timeline

    def load_protocol(self):
        '''
        '''
        protocol = self.load_data('Protocol')
        self.protocol = protocol
        nstim, nreps = protocol.seqnums.shape
        self.nstimuli = nstim
        self.nrepeats = nreps
        self.stimulus_sequence = protocol.seqnums - 1  # change to 0-index

    @property
    def timeline_sample_ratehz(self):
        sample_ratehz = float(1.0/self.timeline.hw[0, 0].samplingInterval)
        return sample_ratehz

    def get_timeline_data(self, event_name):
        '''
        '''
        assert 'neuralFrames' in self.hwevents
        channel_index = self.hwevents.index(event_name)
        ttl = self.timeline.rawDAQData[:, channel_index]
        times = self.timeline.rawDAQTimestamps[0]
        # shift frame time to middle of acquisition
        return times, ttl

    def get_fusi_acq_times(self, shift=0.5):
        '''
        '''
        assert 'neuralFrames' in self.hwevents
        channel_index = self.hwevents.index('neuralFrames')
        ttl = self.timeline.rawDAQData[:, channel_index]
        nonzero = np.diff(ttl).nonzero()[0]
        frame_times = self.timeline.rawDAQTimestamps[0, nonzero]
        # shift frame time to middle of acquisition
        offset = 0 if shift == 0 else 1.0/shift
        fusi_times = (frame_times + self.nBFpframe /
                      offset/self.bfrate).squeeze()
        return fusi_times

    def load_doppler(self, trim_end=2):
        '''
        '''
        doppler = self.load_data('fus')
        frames = doppler.frames

        # store sampling info
        self.nBFpframe = 180.0 if not hasattr(
            doppler, 'nBFPerFrame') else doppler.nBFPerFrame
        self.bfrate = 1.0/doppler.dtBF

        # get time stamps
        acq_times = self.get_fusi_acq_times(shift=0.5)

        # trim data
        if trim_end:
            acq_times = acq_times[:-trim_end]
            nframes = len(acq_times)
            nskipped_frames = doppler.softTimes.shape[0] - nframes
            frames = frames[..., nskipped_frames:]

        self.fusi_frames = frames
        self.fusi_times = acq_times
        self.xaxis = doppler.xAxis
        self.zaxis = doppler.zAxis
        self.aspect = doppler.zAxis.max() / doppler.xAxis.max()

    def load_stimulus_events(self):
        '''
        '''
        timeline = self.timeline
        event_names = [t[0][0].split()[0]
                       for t in timeline.mpepUDPEvents[:int(timeline.mpepUDPCount)]]
        # start and end events
        start_idx = np.asarray(
            [i for i, t in enumerate(event_names) if 'StimStart' == t])
        end_idx = np.asarray(
            [i for i, t in enumerate(event_names) if 'StimEnd' == t])
        # start and end times
        start_times = timeline.mpepUDPTimes[:int(
            timeline.mpepUDPCount)][start_idx].squeeze()
        end_times = timeline.mpepUDPTimes[:int(
            timeline.mpepUDPCount)][end_idx].squeeze()
        self.stimulus_start_times = np.atleast_1d(start_times)
        self.stimulus_end_times = np.atleast_1d(end_times)

    def get_timeline_object(self):
        '''
        '''
        return self.timeline

    def load_photodiode_events(self, **kwargs):
        '''
        kwargs are passed to bandpass filter
        '''
        assert 'photoDiode' in self.hwevents
        timeline = self.timeline

        channel_index = self.hwevents.index('photoDiode')
        phd = timeline.rawDAQData[:, channel_index]
        sample_rate = float(1.0/timeline.hw[0, 0].samplingInterval)
        self.photod_data = phd
        self.photod_samplerate = sample_rate

        fphd = band_pass_signal(phd, sample_rate, **kwargs)
        self.photod_fdata = fphd

    def load_photod_stimuli(self, daqtimes=None):
        '''
        '''
        if not hasattr(self, 'photod_data'):
            self.load_photodiode_events()

        if not hasattr(self, 'stimulus_start_times'):
            self.load_stimulus_events()

        if daqtimes is None:
            daqtimes = self.timeline.rawDAQTimestamps[0]

        threshold = 0.0
        phd = self.photod_data
        phdfilt = self.photod_fdata
        timeline = self.timeline
        start_times = self.stimulus_start_times
        end_times = self.stimulus_end_times

        phdabove = (phdfilt > threshold).astype(np.float32)
        phdtransitions = np.r_[[0], np.diff(phdabove)]
        print((phdtransitions == -1).sum())

        all_ups = daqtimes[phdtransitions == 1]
        all_downs = daqtimes[phdtransitions == -1]

        raw_transitions = np.r_[[0], np.diff(
            ((phd - phd.mean()) > threshold).astype(np.float32))]
        raw_ups = daqtimes[raw_transitions == 1]
        raw_downs = daqtimes[raw_transitions == -1]

        # align stimuli to photodiode
        ##############################
        filter_offset = 0.016
        phdtimes = []

        for stimidx, (onset, offset) in enumerate(zip(start_times, end_times)):
            first_up = raw_ups[raw_ups > onset][0] - filter_offset
            last_down = raw_downs[raw_downs < offset][-1] + filter_offset
            if 0:
                print(stimidx, onset, first_up, last_down)

            # use filtered signals
            # find the first up event
            up_first_idx = (all_ups > first_up).nonzero()[0][0]
            # find the last down event
            down_last_idx = (all_downs < last_down).nonzero()[0][-1]
            # find the last up event
            up_last_idx = (all_ups < all_downs[down_last_idx]).nonzero()[0][-1]
            # find the first down event
            down_first_idx = (
                all_downs > all_ups[up_first_idx]).nonzero()[0][0]

            print(all_ups[up_first_idx], all_ups[up_last_idx])
            print(all_downs[down_first_idx], all_downs[down_last_idx])
            print()

            # store the onset off-set times per stimulus
            up_times = all_ups[up_first_idx:up_last_idx]
            down_times = all_downs[down_first_idx:down_last_idx]
            phdtimes.append((up_times, down_times))

        phdtimes = np.asarray(phdtimes)

        # more robust to skipped frames
        stimuli_start = np.asarray([t[0][0] for t in phdtimes])
        stimuli_end = np.asarray([t[1][-1] for t in phdtimes])

        # stimulus with frame vectors:
        # [(onset1, offset1), (onset2, offset2), ..., (onsetN, offsetN)]
        if phdtimes.ndim == 3:
            stimulus_frame_times = np.sort(
                phdtimes.reshape(phdtimes.shape[0], -1), axis=-1)
        else:
            stimulus_frame_times = []
            for start_times, end_times in zip(phdtimes[:, 0], phdtimes[:, 1]):
                vec = np.r_[start_times, end_times]
                np.sort(vec)
                stimulus_frame_times.append(vec)
            stimulus_frame_times = np.asarray(stimulus_frame_times)

        self.phd_frame_times = stimulus_frame_times[self.stimulus_sequence]
        self.phd_stim_start = stimuli_start[self.stimulus_sequence]
        self.phd_stim_end = stimuli_end[self.stimulus_sequence]
        self.phd_raw = phdtimes[self.stimulus_sequence]  # new

    def set_frame_markers(self):
        '''
        '''
        import itertools
        frame_markers = np.zeros(
            (self.nstimuli, self.nrepeats, self.fusi_times.shape[0]))

        for idx, (sdx, rdx) in enumerate(itertools.product(range(self.nstimuli), range(self.nrepeats))):
            ontime = self.phd_stim_start[sdx, rdx]
            offtime = self.phd_stim_end[sdx, rdx]
            marker = np.logical_and(self.fusi_times < offtime,
                                    self.fusi_times > ontime)
            print(np.asarray([idx, sdx, rdx, ontime,
                  offtime, offtime-ontime, marker.sum()]))
            frame_markers[sdx, rdx, :] = marker
        self.stimulus_markers = frame_markers


class StimulusInfo(DataStore):
    def __init__(self, *args, **kwargs):
        super(StimulusInfo, self).__init__(*args, **kwargs)
        self.load_protocol()

    def load_timeline(self):
        '''
        '''
        timeline = self.load_data('Timeline')
        HWEVENTS = [t.name[0] for t in timeline.hw[0, 0].inputs[0]]
        self.hwevents = HWEVENTS
        self.timeline = timeline

    def load_protocol(self):
        '''
        '''
        protocol = self.load_data('Protocol')
        self.protocol = protocol
        nstim, nreps = protocol.seqnums.shape
        self.nframes_per_swipe = 499
        self.monitor_fps = 60
        self.nswipes = 6
        self.nstimuli = nstim
        self.nrepeats = nreps
        self.stimulus_sequence = protocol.seqnums - 1  # change to 0-index

    def load(self):
        self.load_software_times()
        self._load_photodiode_data()
        self.load_hardware_times()
        self.guess_swipe_stimulus_times()

    def load_software_times(self):
        '''Stimulus events from software
        '''
        self.load_timeline()
        timeline = self.timeline
        event_names = [t[0][0].split()[0]
                       for t in timeline.mpepUDPEvents[:int(timeline.mpepUDPCount)]]
        # start and end events
        start_idx = np.asarray(
            [i for i, t in enumerate(event_names) if 'StimStart' == t])
        end_idx = np.asarray(
            [i for i, t in enumerate(event_names) if 'StimEnd' == t])
        # start and end times
        start_times = timeline.mpepUDPTimes[:int(
            timeline.mpepUDPCount)][start_idx].squeeze()
        end_times = timeline.mpepUDPTimes[:int(
            timeline.mpepUDPCount)][end_idx].squeeze()

        self.software_start_times = np.atleast_1d(start_times)
        self.software_end_times = np.atleast_1d(end_times)

    def _load_photodiode_data(self, **kwargs):
        '''
        kwargs are passed to bandpass filter
        '''
        assert 'photoDiode' in self.hwevents
        timeline = self.timeline

        channel_index = self.hwevents.index('photoDiode')
        phd = timeline.rawDAQData[:, channel_index]
        sample_rate = float(1.0/timeline.hw[0, 0].samplingInterval)
        fphd = band_pass_signal(phd, sample_rate, **kwargs)

        self.photod_data = phd
        self.photod_samplerate = sample_rate
        self.photod_fdata = fphd

    def load_hardware_times(self):
        '''
        '''
        # if not hasattr(self, 'photod_data'):
        #     self.load_photodiode_events()

        # if not hasattr(self, 'stimulus_start_times'):
        #     self.load_stimulus_events()

        threshold = 0.0
        phd = self.photod_data
        phdfilt = self.photod_fdata
        timeline = self.timeline
        start_times = self.software_start_times
        end_times = self.software_end_times

        phdabove = (phdfilt > threshold).astype(np.float32)
        phdtransitions = np.r_[[0], np.diff(phdabove)]
        print((phdtransitions == -1).sum())

        all_ups = timeline.rawDAQTimestamps[0][phdtransitions == 1]
        all_downs = timeline.rawDAQTimestamps[0][phdtransitions == -1]

        raw_transitions = np.r_[[0], np.diff(
            ((phd - phd.mean()) > threshold).astype(np.float32))]
        raw_ups = timeline.rawDAQTimestamps[0][raw_transitions == 1]
        raw_downs = timeline.rawDAQTimestamps[0][raw_transitions == -1]

        # align stimuli to photodiode
        ##############################
        filter_offset = (1./60)  # 0.016
        phdtimes = []

        for stimidx, (onset, offset) in enumerate(zip(start_times, end_times)):
            first_up = raw_ups[raw_ups > onset][0] - filter_offset
            last_down = raw_downs[raw_downs < offset][-1] + filter_offset
            if 0:
                print(stimidx, onset, first_up, last_down)

            # use filtered signals
            # find the first up event
            up_first_idx = (all_ups > first_up).nonzero()[0][0]
            # find the last down event
            down_last_idx = (all_downs < last_down).nonzero()[0][-1]
            # find the last up event
            up_last_idx = (all_ups < all_downs[down_last_idx]).nonzero()[0][-1]
            # find the first down event
            down_first_idx = (
                all_downs > all_ups[up_first_idx]).nonzero()[0][0]

            print(all_ups[up_first_idx], all_ups[up_last_idx])
            print(all_downs[down_first_idx], all_downs[down_last_idx])
            print()

            # store the onset off-set times per stimulus
            up_times = all_ups[up_first_idx:up_last_idx]
            down_times = all_downs[down_first_idx:down_last_idx]
            phdtimes.append((up_times, down_times))

        phdtimes = np.asarray(phdtimes)

        # more robust to skipped frames
        stimuli_start = np.asarray([t[0][0] for t in phdtimes])
        stimuli_end = np.asarray([t[1][-1] for t in phdtimes])

        # stimulus with frame vectors:
        # [(onset1, offset1), (onset2, offset2), ..., (onsetN, offsetN)]
        if phdtimes.ndim == 3:
            stimulus_frame_times = np.sort(
                phdtimes.reshape(phdtimes.shape[0], -1), axis=-1)
        else:
            stimulus_frame_times = []
            for start_times, end_times in zip(phdtimes[:, 0], phdtimes[:, 1]):
                vec = np.r_[start_times, end_times]
                print(vec.shape, start_times.shape, end_times.shape)
                vec = np.sort(vec)
                stimulus_frame_times.append(vec)
            stimulus_frame_times = np.asarray(stimulus_frame_times)

        # re-order to keep repeat and stimulus identitites
        self.phd_frame_times = stimulus_frame_times[self.stimulus_sequence]
        self.phd_stim_start = stimuli_start[self.stimulus_sequence]
        self.phd_stim_end = stimuli_end[self.stimulus_sequence]
        self.phd_raw = phdtimes[self.stimulus_sequence]

        # get swipe events
        ##############################
        swipes = []
        for idx, (sdx, rdx) in enumerate(itertools.product(range(self.nstimuli), range(self.nrepeats))):
            times = self.phd_raw[sdx, rdx]
            diff = times[1] - times[0]
            stats = np.asarray([diff.min(), diff.mean(), diff.max()])
            print(idx, sdx, rdx, times[0].shape,
                  times[1].shape, times[1][-1]-times[0][0], stats)
            # nphotoevents = (self.frames_per_swipe*self.nswipes)/2
            times = np.r_[times[0], times[1]]
            times = np.sort(times)
            times = np.array_split(times, self.nswipes)
            # swipes
            swipes.append(times)
        swipes = np.asarray(swipes).reshape(self.nstimuli, self.nrepeats, -1)
        self.phd_swipes = swipes

    def guess_swipe_stimulus_times(self, stimid=0):
        swipes = np.zeros((self.nstimuli, self.nrepeats,
                          self.nswipes, self.nframes_per_swipe))
        for idx, (sdx, rdx) in enumerate(itertools.product(range(self.nstimuli), range(self.nrepeats))):
            times = self.phd_raw[sdx, rdx]
            diff = times[1] - times[0]
            stats = np.asarray([diff.min(), diff.mean(), diff.max()])
            print(idx, sdx, rdx, times[0].shape,
                  times[1].shape, times[1][-1]-times[0][0], stats)
            # nphotoevents = (self.frames_per_swipe*self.nswipes)/2
            times = np.arange(self.nframes_per_swipe*self.nswipes,
                              dtype=np.float32)/self.monitor_fps + times[0][0]
            times = np.array_split(times, self.nswipes)
            # swipes
            swipes[sdx, rdx] = times
        swipes = np.asarray(swipes)
        self.hardcoded_times = swipes


if 0:
    ds = DataStore('CR017', date=(2019, 11, 13))
    ds = ExperimentData('CR017', date=(2019, 11, 13))
