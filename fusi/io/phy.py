'''Code to interact with Phy output.

Anwar O. Nunez-Elizalde (Dec 2019).
'''
import os
import re
import json
import pathlib

from glob import glob

import numpy as np

import fusi.utils


##############################
# helper functions
##############################

def sanitize_string2attribute(path):
    '''
    '''
    return re.sub(r'[^\w]', '', path)

def str2bool(string):
    '''Convert a `string` to a `bool` type
    '''
    val = True if string in ['True', 'true'] else string
    val = False if string in ['False', 'false'] else string
    val = None if string in ['None', 'none'] else string
    return val


def can_bool(string):
    '''Test whether `string` can map to `bool`
    '''
    booleables = ['True', 'true',
                  'False', 'false',
                  'None', 'none']
    return string in booleables


def yield_table_columns(textfile, delimiter='\t', dtype='S'):
    '''Yield (columns, data) tuples from a table stored as a TXT file.
    '''
    # Load tab-delimited table
    table = np.loadtxt(textfile, delimiter=delimiter, dtype=dtype)
    column_names = [t.decode() for t in table[0]]

    for colidx, column_name in enumerate(column_names):
        column_data = table[1:, colidx]
        try:
            column_data = np.asarray(column_data, dtype=np.float)
            if np.allclose(column_data, np.asarray(column_data, dtype=np.int)):
                column_data = column_data.astype(np.int)
        except:
            # must be string
            column_data = [t.decode() for t in table[1:,colidx]]
            # maybe is full of booleans?
            if np.alltrue([can_bool(t) for t in column_data]):
                column_data = [str2bool(t) for t in column_data]
            # cast to np.ndarray
            column_data = np.asarray(column_data)
        yield column_name, column_data


def load_phy_params(path, params_flname='params.py'):
    '''Load PHY `params.py` as a dictionary
    '''
    # Parse phy parameters
    params = fusi.utils.Bunch()

    flname = os.path.join(path, params_flname)
    if not os.path.exists(flname):
        raise IOError(f'File does not exist: {flname}')

    with open(flname, 'r') as fl:
        # Parse the file. Do not evaluate.
        for line in fl.readlines():
            key, val = line.replace(' ','').strip().split('=')
            try:
                # Handle numerical values
                val = float(val)
                val = int(val) if val == int(val) else val
            except:
                # Handle other values
                val = re.sub("['\"]", '', val).strip() # remove quotes
                # Try to map to boolean
                val = str2bool(val) if can_bool(val) else val
            params[key] = val
    return params


##################################################
# objects for interacting with PHY output
##################################################


class EphysFolder(object):
    '''Searches for things an ephys folder might contain.
    '''
    def __init__(self, path):
        '''
        '''
        path_object = pathlib.Path(path)
        self.__path__ = str(path_object)
        self.__pathobj__ = path_object
        self.__iterables__ = []

    def __name__(self, prefix='', suffix=''):
        name = re.sub(r'[^\w]', '', self.__pathobj__.stem)
        return '%s%s%s'%(prefix, name, suffix)

    @property
    def __niterable__(self):
        return len(self.__iterables__)

    def search(self, pattern, match_index=False):
        '''
        '''
        matches = list(self.__pathobj__.glob(pattern))
        flname = str(matches[0]) if len(matches)==1 else None

        info = (pattern, self.__pathobj__)
        if len(matches) == 0:
            print('No matches for "%s" found in %s'%info)
        elif len(matches) > 1 and match_index is False:
            print('Multiple matches for "%s" found in %s\n'%info,matches)
        elif len(matches) > 1 and match_index:
            # user selection
            flname = str(matches[match_index])
        return flname

    def get_metadata(self, metadata_file='recording_info.json'):
        '''
        '''
        info_path = self.search(metadata_file)
        if info_path:
            with open(str(info_path), 'r') as flo:
                content = json.load(flo)
            return content

    def get_nidq_path(self, pattern='*nidq.bin'):
        return self.search(pattern)

    def get_ap_path(self, pattern='*ap.bin'):
        return self.search(pattern)

    def get_lfp_path(self, pattern='*lf.bin'):
        return self.search(pattern)

    def __iter__(self):
        self.__counter__ = 0
        return self

    def __next__(self):
        if self.__counter__ >= self.__niterable__:
            raise StopIteration
        else:
            attr = getattr(self, self.__iterables__[self.__counter__])
            self.__counter__ += 1
            return attr


class PhyContent(object):
    '''
    '''
    def __init__(self, flname):
        path_object = pathlib.Path(flname)
        self.__ext__ = path_object.suffix
        self.__path__ = str(path_object)
        self.__pathobj__ = path_object

    @property
    def __name__(self):
        return self.__pathobj__.stem

    def __call__(self):
        '''Load the content.
        '''
        self.__load__()

    def __load__(self):
        '''Load the content and attach it to the object
        '''
        fl = self.__path__

        if self.__ext__ == '.npy':
            # Load numpy data
            data = np.load(fl)
            setattr(self, 'data', data)
        elif self.__ext__ in ['.tsv', '.csv']:
            table_delimiters = {'.tsv' : '\t',
                                '.csv' : ','}
            delim = table_delimiters[self.__ext__]
            column_generator = yield_table_columns(fl, delimiter=delim, dtype='S')
            for colnum, (column_name, column_data) in enumerate(column_generator):
                setattr(self, column_name, column_data)
        else:
            raise ValueError('Unknown extension: %s'%self.__ext__)

    def __repr__(self):
        info = (__name__, type(self).__name__, self.__pathobj__.stem)
        return '<%s.%s.%s>'%info


class ProbeHandler(EphysFolder):
    '''Assumed directory structures:

    path/<probe1>/spike_clusters.npy
    path/<probe2>/spike_clusters.npy
    path/recording_info.json

    object.probe1.<phy>
    object.probe2.<phy>
    '''

    def __init__(self, path):
        '''
        '''
        super(type(self), self).__init__(path)

        # Find all files output by Phy
        path_object = pathlib.Path(path)
        files = (list(path_object.glob('*.npy')) +
                 list(path_object.glob('*.tsv')))

        iterables = []
        for fl in files:
            name = '%s_%s'%(fl.suffix.strip('.'), fl.stem)
            phy_object = PhyContent(str(fl))
            setattr(self, name, phy_object)
            iterables += [name]

        # load PHY parameters
        try:
            params = load_phy_params(path)
            setattr(self, 'params', params)
        except IOError:
            # Parameters not available
            pass

        self.__path__ = path
        self.__pathobj__ = path_object
        self.__iterables__ = iterables
    def __repr__(self):
        return '<%s.%s [%s]>'%(__name__, type(self).__name__, self.__pathobj__.stem)

    def get_params_flname(self, flname='params.py'):
        '''
        '''
        fl = self.__pathobj__/flname
        assert fl.exists()
        return fl

    def extract_waveforms(self, clusterid):
        '''
        '''
        from phylib.io.model import load_model

        # First, we load the TemplateModel.
        model = load_model(self.get_params_flname())  # first argument: path to params.py

        # We get the waveforms of the cluster.
        waveforms = model.get_cluster_spike_waveforms(clusterid)
        print('n_spikes=%i, n_samples=%i, n_channels_loc=%i'%waveforms.shape)

        # We get the channel ids where the waveforms are located.
        channel_ids = model.get_cluster_channels(clusterid)
        return channel_ids, waveforms

    def extract_mean_waveform(self, clusterid):
        '''
        '''
        from phylib.io.model import load_model
        # First, we load the TemplateModel.
        model = load_model(self.get_params_flname())  # first argument: path to params.py
        odict = model.get_cluster_mean_waveforms(clusterid)
        snr = np.abs(odict['mean_waveforms']).sum(0).max()
        biggest = np.abs(odict['mean_waveforms']).sum(0).argmax()
        mean_waveform = odict['mean_waveforms'][:, biggest]
        return snr, mean_waveform[...,None]

    @property
    def probe_id(self, pattern='imec', splitter='_'):
        '''
        '''
        if pattern in self.__name__:
            parts = self.__name__.split(splitter)
            parts = [part for part in parts if pattern in part]
            if len(parts) == 1:
                return parts[0]

    @property
    def probe_name(self):
        '''
        '''
        assert self.probe_id[:-1] == 'imec'
        return 'probe%02i'%int(self.probe_id[-1])


    @property
    def __name__(self):
        name = super(type(self), self).__name__(prefix='probe_')
        return name

    def get_digital_channel(self, channel_index=-1):
        '''Load the digital channel from the binary AP ephys data.
        '''
        from fusi.io import spikeglx
        path = self.get_ap_path()
        print('Loading digital signal from: %s'%path)
        probe_memmap = spikeglx.get_spikeglx_memmap(path)
        # really slow:
        probe_block_npsync = np.ascontiguousarray(probe_memmap[channel_index, :],
                                                  dtype=probe_memmap.dtype)
        # really fast
        probe_block_npsync[probe_block_npsync != 0] = 1
        return probe_block_npsync

    def get_spikeglx_metadata(self):
        '''Load SpikeGLX metadata for this probe
        '''
        from fusi.io import spikeglx
        path = self.get_ap_path()
        return spikeglx.read_metadata(path)

    @property
    def sample_ratehz(self):
        '''Sample rate [Hz] read from SpikeGLX file
        '''
        from fusi.io import spikeglx
        metadata = spikeglx.read_metadata(self.get_ap_path())
        return metadata['imSampRate']

class RecordingHandler(EphysFolder):
    '''
    '''
    def __init__(self, path):
        '''Assumed directory structures:

        path/<probe1>/spike_clusters.npy
        path/<probe2>/spike_clusters.npy
        path/recording_info.json

        object.probe1.<phy>
        object.probe2.<phy>
        '''
        super(type(self), self).__init__(path)
        path_object = pathlib.Path(path)
        matches = list(path_object.glob('*/spike_clusters.npy'))

        if len(matches) == 0:
            raise ValueError('Nothing found! %s'%path)

        iterables = []
        for match in matches:
            root_directory = match.parent
            name = re.sub(r'[^\w]', '', root_directory.stem)
            name = 'probe_%s'%str(name)
            handler = ProbeHandler(str(root_directory))
            setattr(self, name, handler)
            iterables += [name]

        self.__probes__ = [match.parent for match in matches]
        self.__path__ = str(path_object)
        self.__pathobj__ = path_object
        self.__iterables__ = iterables

    def __repr__(self):
        info = (__name__, type(self).__name__, str(self.__pathobj__), len(self.__probes__))
        return '<%s.%s source:%s [%i probes]>'%info

    def get_probe(self, num=0):
        '''
        '''
        return getattr(self, self.__iterables__[num])

    @property
    def __name__(self):
        name = super(type(self), self).__name__(prefix='recording_')
        return name


class SessionHandler(EphysFolder):
    '''
    '''
    def __init__(self, path):
        '''Assumed directory structures:

        path/<recording1>/<probe1>/spike_clusters.npy
        path/<recording1>/<probe2>/spike_clusters.npy
        path/<recording1>/recording_info.json
        path/<recording2>/<probe1>/spike_clusters.npy
        path/<recording2>/<probe2>/spike_clusters.npy
        path/<recording2>/recording_info.json
        path/<recording3>/<probe1>/spike_clusters.npy
        path/<recording3>/<probe2>/spike_clusters.npy


        object.recording_<recording1>.probe_<probe1>.<phy>
        object.recording_<recording1>.probe_<probe2>.<phy>
        object.recording_<recording2>.probe_<probe1>.<phy>
        object.recording_<recording2>.probe_<probe2>.<phy>
        object.recording_<recording3>.probe_<probe1>.<phy>
        object.recording_<recording3>.probe_<probe2>.<phy>
        '''
        super(type(self), self).__init__(path)
        path_object = pathlib.Path(path)
        matches = list(path_object.glob('*/*/spike_clusters.npy'))

        # These are also path objects
        probes = [match.parent for match in matches]
        recordings = list(set([probe.parent for probe in probes]))

        if len(matches) == 0:
            raise ValueError('Nothing found! %s'%path)

        iterables = []
        for recording_path in recordings:
            name = re.sub(r'[^\w]', '', recording_path.stem)
            name = 'recording_%s'%str(name)
            handler = RecordingHandler(str(recording_path))
            setattr(self, name, handler)
            iterables += [name]

        self.__probes__ = probes
        self.__recordings__ = recordings
        self.__path__ = path
        self.__pathobj__ = path_object
        self.__iterables__ = iterables

    def __repr__(self):
        info = (__name__, type(self).__name__, str(self.__pathobj__),
                len(self.__recordings__), len(self.__probes__))
        return '<%s.%s source:%s [%i recording(s), %i probes]>'%info

    def get_recording(self, num=0):
        '''
        '''
        return getattr(self, self.__iterables__[num])

    @property
    def __name__(self):
        name = super(type(self), self).__name__(prefix='session_')
        return name



# def extract_waveforms(cluster_id):
    # from phylib.io.model import load_model
    # # from phylib.utils.color import selected_cluster_color
    # # First, we load the TemplateModel.
    # model = load_model(sys.argv[1])  # first argument: path to params.py
    # # We obtain the cluster id from the command-line arguments.
    # cluster_id = int(sys.argv[2])  # second argument: cluster index

#     # We get the waveforms of the cluster.
#     waveforms = model.get_cluster_spike_waveforms(cluster_id)
#     n_spikes, n_samples, n_channels_loc = waveforms.shape
#
#     # We get the channel ids where the waveforms are located.
#     channel_ids = model.get_cluster_channels(cluster_id)


if __name__ == '__main__':
    from fusi.config import DATA_ROOT
    path = f'{DATA_ROOT}/CR017/2019-11-13/ephys/2019-11-13_CR017_g0/2019-11-13_CR017_g0_imec0'
    phandler = PhyHandler(path)
    # times = phandler.spike_times.data.astype(np.float64)/phandler.params.sample_rate
