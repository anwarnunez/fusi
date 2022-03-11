'''
Data handler that relies on INI logs and `fusi.io`.

Anwar O. Nunez-Elizalde (Jan, 2019)
'''
import os
import pathlib
from glob import glob
from pprint import pprint
from collections import OrderedDict

import numpy as np

from fusi import misc, utils as futils
from fusi.io import spikeglx, phy, logs, sync, cxlabexp
from fusi.extras import readers
import fusi.config


def hdf_load(flname, *args, **kwargs):
    '''
    '''
    local_path = misc.uri_convert_uri2local(str(flname))
    return readers.hdf_load(local_path, *args, **kwargs)


def mk_hash_call(func_kwargs):
    '''
    '''
    import hashlib
    _ = func_kwargs.pop('self')
    _ = func_kwargs.pop('recache')
    call_signature = str(func_kwargs).encode()
    print(call_signature)
    call_hash = hashlib.sha512(call_signature).hexdigest()
    return call_hash


def params2cache(kwargs):
    '''
    '''
    thash = mk_hash_call(kwargs)
    outfl = os.path.join('/store/fast/scratch', str(thash))
    return pathlib.Path(outfl)


EXPERIMENTS = [('CR017', (2019, 11, 13)),
               ('CR017', (2019, 11, 14)),
               ('CR019', (2019, 11, 26)), 
               ('CR019', (2019, 11, 27)),
               ('CR020', (2019, 11, 20)),
               ('CR020', (2019, 11, 21)),
               ('CR020', (2019, 11, 22)),
               ('CR022', (2020, 10, 7)),
               ('CR022', (2020, 10, 11)),
               ('CR024', (2020, 10, 29)),
               ]


##############################
# helper functions
##############################
class MetaSession(object):
    '''
    '''

    def __init__(self,
                 subject_name,
                 session_name,
                 root=None,
                 verbose=False):
        '''
        '''
        if root is None:
            root = fusi.config.DATA_ROOT
        root = pathlib.Path(str(root)).joinpath(subject_name)
        session_path = root.joinpath(session_name)

        self.root = root
        self.subject_name = subject_name
        self.session_name = session_name
        self.session_path = session_path
        self.verbose = verbose

        if not root.exists():
            raise IOError('Invalid path: %s' % root)
        if not session_path.exists():
            raise IOError('Invalid path: %s' % session_path)

        self.log = None
        self.load_session_log(verbose=verbose)

    def ls(self, pattern):
        '''
        '''
        results = sorted(list(self.session_path.glob(pattern)))
        return results

    @property
    def experiment_mapper_slices2blocks(self):
        '''A dict mapping slices to session blocks
        '''
        session_info = self.log_load_section('experiment')
        slice_blocks = self.log_load_section('fusi', 'mapping_block2slices')
        fusi_blocks = session_info['matched_fusi_blocks']
        assert len(fusi_blocks) == len(slice_blocks)

        # Get the blocks corresponding for each slice
        all_slice_indeces = np.unique(slice_blocks)
        # remove nans
        all_slice_indeces = all_slice_indeces[np.logical_not(
            np.isnan(all_slice_indeces))]
        # check they are indeces
        assert np.allclose(np.asarray(all_slice_indeces,
                           dtype=np.int), all_slice_indeces)
        all_slice_indeces = np.asarray(all_slice_indeces, dtype=np.int)
        slice_blocks_dict = {slice_num: fusi_blocks[slice_blocks == slice_num]
                             for slice_num in all_slice_indeces}
        return slice_blocks_dict

    @property
    def experiment_mapper_blocks2slices(self):
        slice_blocks = self.experiment_mapper_slices2blocks
        blocks2slice = {}
        for sl, blocks in slice_blocks.items():
            for block in blocks:
                blocks2slice[block] = sl
        return blocks2slice

    @property
    def experiment_blocks(self):
        '''Get all blocks
        '''
        session_info = self.log_load_section('experiment')
        blocks = session_info['block_numbers']
        blocks = blocks[np.logical_not(np.isnan(blocks))]
        if 'valid_blocks' in session_info:
            blocks = session_info['valid_blocks']
        if 'invalid_blocks' in session_info:
            bad_blocks = session_info['invalid_blocks']
            blocks = np.asarray([t for t in blocks if t not in bad_blocks])
        return blocks.astype(np.int)

    @property
    def experiment_fusi_blocks(self):
        '''
        '''
        session_info = self.log_load_section('experiment')
        blocks = self.log_load_section('experiment', 'matched_fusi_blocks')
        blocks = blocks[np.logical_not(np.isnan(blocks))]
        if 'valid_blocks' in session_info:
            blocks = session_info['valid_blocks']
        if 'invalid_blocks' in session_info:
            bad_blocks = session_info['invalid_blocks']
            blocks = np.asarray([t for t in blocks if t not in bad_blocks])
        return blocks.astype(np.int)

    def fusi_blocks_iterator(self, blocks=None):
        '''
        '''
        if blocks is None:
            blocks = self.experiment_fusi_blocks

        for block in blocks:
            yield MetaBlock(self.subject_name, self.session_name, block)

    @property
    def fusi_nslices(self):
        return len(self.experiment_mapper_slices2blocks)

    def fusi_slice_position(self, slice_number):
        '''Get position of slice in [mm]
        '''
        info = self.log_load_section('fusi')
        slices_mm = info['slice_positions']
        return slices_mm[slice_number]

    def experiment_slice2blocks(self, slice_number):
        '''Get the fUSi blocks corresponding to the slice of interest
        '''
        mapper = self.experiment_mapper_slices2blocks
        return mapper[slice_number]

    @property
    def experiment_tasks(self):
        return self.log_load_section('experiment', 'tasks')

    def experiment_get_task_blocks(self, task_name):
        '''
        '''
        info = self.log_load_section('experiment')
        assert task_name in info['tasks']
        return np.asarray(info['blocks_%s' % task_name])

    def fusi_get_slice_data(self, slice_number,
                            fusi_mask=None,
                            fusi_preproc=dict(dt_ms=50,
                                              window_ms=150,
                                              svddrop=5,
                                              freq_cutoffhz=15,
                                              roi_name=None,
                                              mirrored=False),
                            trim_raw=dict(trim_beg=5, trim_end=5),
                            trim_post_detrend=dict(trim_beg=10, trim_end=10),
                            outbrain_nuisance=None,
                            temporal_detrend=dict(ftype='sg',
                                                  window_size_sec=120,
                                                  polyorder=3),
                            spatial_smoothing=None,
                            normalization='pctchg'):
        '''
        '''
        fusi_slice_blocks = self.experiment_slice2blocks(slice_number)
        print('Working on slice %i' % slice_number, fusi_slice_blocks)

        slice_data = []
        slice_times = []
        for bdx, block_number in enumerate(fusi_slice_blocks):
            fusi_times, fusi_data = self.fusi_get_block_data(
                block_number,
                fusi_mask=fusi_mask,
                fusi_preproc=fusi_preproc,
                trim_raw=trim_raw,
                trim_post_detrend=trim_post_detrend,
                outbrain_nuisance=outbrain_nuisance,
                temporal_detrend=temporal_detrend,
                spatial_smoothing=spatial_smoothing,
                normalization=normalization)

            # store
            slice_data.append(fusi_data)
            slice_times.append(fusi_times)

        slice_data = np.vstack(slice_data)
        return slice_times, slice_data

    def fusi_get_multiblock_data(self, block_numbers,
                                 fusi_mask=None,
                                 fusi_preproc=dict(dt_ms=50,
                                                   window_ms=150,
                                                   svddrop=5,
                                                   freq_cutoffhz=15,
                                                   roi_name=None,
                                                   mirrored=False),
                                 trim_raw=dict(trim_beg=5, trim_end=5),
                                 trim_post_detrend=dict(
                                     trim_beg=10, trim_end=10),
                                 outbrain_nuisance=None,
                                 temporal_detrend=dict(ftype='sg',
                                                       window_size_sec=120,
                                                       polyorder=3),
                                 spatial_smoothing=None,
                                 normalization='pctchg'):
        '''
        fusi_mask : A mask to apply to the voxels of the fUSi slice
        '''
        blocks_data = []
        blocks_times = []
        for bdx, block_number in enumerate(block_numbers):
            fusi_times, fusi_data = self.fusi_get_block_data(
                block_number,
                fusi_mask=fusi_mask,
                fusi_preproc=fusi_preproc,
                trim_raw=trim_raw,
                trim_post_detrend=trim_post_detrend,
                outbrain_nuisance=outbrain_nuisance,
                temporal_detrend=temporal_detrend,
                spatial_smoothing=spatial_smoothing,
                normalization=normalization)

            # store
            blocks_data.append(fusi_data)
            blocks_times.append(fusi_times)

        blocks_data = np.vstack(blocks_data)
        return blocks_times, blocks_data

    def fusi_get_block_data(self, block_number,
                            fusi_mask=None,
                            fusi_preproc=dict(dt_ms=50,
                                              window_ms=150,
                                              svddrop=5,
                                              freq_cutoffhz=15,
                                              roi_name=None,
                                              mirrored=False),
                            trim_raw=dict(trim_beg=5, trim_end=5),
                            trim_post_detrend=dict(trim_beg=10, trim_end=10),
                            outbrain_nuisance=None,
                            temporal_detrend=dict(ftype='sg',
                                                  window_size_sec=120,
                                                  polyorder=3),
                            spatial_smoothing=None,
                            normalization='pctchg',
                            ):
        '''
        fusi_mask : A mask to apply to the voxels of the fUSi slice
        outbrain_nuisance = dict(npcs=3,r2thresh=0.1),
        spatial_smoothing=dict(sigma_um=20),
        '''
        fusi_dt_sec = fusi_preproc['dt_ms']/1000.0

        subject_block = MetaBlock(self.subject_name,
                                  self.session_name,
                                  int(block_number))
        fusi_times, fusi_data = subject_block.fusi_get_data(**fusi_preproc)

        # trim before detrending
        ##############################
        if trim_raw is not None:
            times_valid = np.logical_and(fusi_times > (fusi_times.min()
                                                       + trim_raw['trim_beg']),
                                         fusi_times < (fusi_times.max()
                                                       - trim_raw['trim_end']))
            fusi_times = fusi_times[times_valid]
            fusi_data = fusi_data[times_valid]
            fusi_shape = fusi_data.shape[:]

        # outbrain_nuisance
        ##############################
        if outbrain_nuisance is not None:
            from scipy import linalg as LA
            from tikreg import models, utils as tikutils
            outbrain_npcs = outbrain_nuisance['npcs']
            r2thresh = outbrain_nuisance['r2thresh']

            # get temporal PCs from outside brain mask
            outsidebrain_mask = subject_block.fusi_get_outsidebrain_mask()
            nuisance_xdata = LA.svd(fusi_data[:, outsidebrain_mask],
                                    full_matrices=False)[0][:, :outbrain_npcs]
            # fit OLS model and obtain predictions
            nuisance_ypreds = models.olspred(
                nuisance_xdata, fusi_data.reshape(fusi_data.shape[0], -1))
            # compute prediction accuracy
            nuisance_r2 = tikutils.columnwise_rsquared(
                nuisance_ypreds, fusi_data.reshape(fusi_data.shape[0], -1))
            # voxels to clean
            nuisance_mask = (nuisance_r2 > r2thresh).reshape(
                fusi_shape[1:])  # > 1%
            # remove variance explained from outside brain
            fusi_data[:, nuisance_mask] = np.nan_to_num(
                (fusi_data[:, nuisance_mask] -
                 nuisance_ypreds.reshape(fusi_data.shape)[:, nuisance_mask]))

        # flatten and mask data
        ##############################
        if fusi_mask is not None:
            assert isinstance(fusi_mask, np.ndarray)
            fusi_data = fusi_data[:, fusi_mask]
        else:
            fusi_data = fusi_data.reshape(fusi_data.shape[0], -1)

        # temporal detrend
        ##############################
        if temporal_detrend is not None:
            window_size = int(temporal_detrend['window_size_sec']/fusi_dt_sec)
            window_size += (window_size - 1) % 2  # make odd number
            yfull = futils.temporal_filter(fusi_data - fusi_data.mean(0),  # zero mean
                                           ftype=temporal_detrend['ftype'],
                                           window_size=window_size,
                                           polyorder=temporal_detrend['polyorder'])
            # need mean for % signal change
            yfull += fusi_data.mean(0)

        # trim after detrending
        ##############################
        if trim_raw is not None:
            times_valid = np.logical_and(fusi_times > (fusi_times.min()
                                                       + trim_post_detrend['trim_beg']),
                                         fusi_times < (fusi_times.max()
                                                       - trim_post_detrend['trim_end']))
            fusi_times = fusi_times[times_valid]
            fusi_data = fusi_data[times_valid]
            fusi_shape = fusi_data.shape[:]

        # smoothing
        ##############################
        if spatial_smoothing is not None:
            from scipy import ndimage
            sigma_um = spatial_smoothing['sigma_um']
            um_hsize, um_vsize = np.asarray(
                subject_block.fusi_image_pixel_mm)*1000
            sigma_hpix = sigma_um/um_hsize
            hsigma, vsigma = (sigma_hpix, sigma_hpix /
                              subject_block.fusi_aspect_ratio)
            sigmas = (vsigma, hsigma)
            print(
                'Smoothing: %0.02d[um]. pixels (vert, horiz): ' % sigma_um, sigmas)

            smooth_data = np.zeros_like(fusi_data)
            for image_index in range(fusi_data.shape[0]):
                img = fusi_data[image_index]
                sim = ndimage.gaussian_filter(img, sigma=sigmas)
                smooth_data[image_index] = sim
            fusi_data = smooth_data

        # normalize signals
        ##############################
        if normalization == 'pctchg':
            fusi_data = futils.pixelwise_pctsignalchange(fusi_data)
        elif normalization == 'zscore':
            fusi_data = zscore(fusi_data)

        return fusi_times, fusi_data

    def cortexlab_mk_filename(self, filename, subfolder=None, mkdir=False):
        date_txt = misc.date_tuple2cortexlab(self.date_tuple)
        flname = '{date}_{subject}_{fl}'.format(date=date_txt,
                                                subject=self.subject_name,
                                                fl=filename)
        if subfolder is None:
            outpath = self.session_path
        else:
            outpath = self.session_path.joinpath(subfolder)
            if mkdir:
                outpath.mkdir(exist_ok=True)

        return outpath.joinpath(flname)

    def __str__(self):
        return '%s %s' % (self.subject_name, self.session_name)

    def __repr__(self):
        info = (__name__, type(self).__name__,
                self.subject_name, self.session_name)
        return '<%s.%s [%s: session=%s]>' % info

    @property
    def date_tuple(self):
        date = misc.date_cortexlab2tuple(self.session_name)
        return date

    @property
    def subject_sessions(self):
        return list(self.root.glob('20??-??-??'))

    @property
    def session_blocks(self):
        blocks = []
        maxdigits = 2
        for num in range(1, maxdigits+1):
            blocks += list(self.session_path.glob('[0-9]'*num))
        # sort by block number
        blocks = sorted(blocks, key=lambda x: int(x.stem))
        return blocks

    def load_session_log(self, verbose=False):
        '''
        '''
        pattern = '{session_path}/{year}-{month}-{day}_{subject}.ini'
        yyyy, mm, dd = misc.date_tuple2yyyymmdd(self.date_tuple)
        flname = pattern.format(session_path=self.session_path,
                                subject=self.subject_name,
                                year=yyyy,
                                month=mm,
                                day=dd)
        try:
            log = logs.read_experiment_log(flname, verbose=verbose)
        except IOError:
            print('Log file not available: %s' % flname)
            log = None
        self.log = log

    def log_load_section(self, section_name, field=None):
        contents = self.log[section_name]
        # Convert the contents into correct datatypes
        parsed_contents = {}
        for key in contents.keys():
            if (field is not None) and (key != field):
                continue

            value = contents[key]
            value = misc.convert_string2data(value)

            if isinstance(value, list):
                # convert numerical lists to arrays
                isnumeric = [not isinstance(val, str) for val in value]
                if np.alltrue(isnumeric):
                    value = np.asarray(value)
            if isinstance(value, str):
                # Dynamic root path
                if '{dataset_full_path}' in value:
                    value = value.format(
                        dataset_full_path=fusi.config.DATA_ROOT)

            # store parsed result
            parsed_contents[key] = value

        # Only return what is requested
        if field is not None:
            if field not in parsed_contents:
                print('Not found: %s/%s' % (section_name, field))
            else:
                parsed_contents = parsed_contents[field]
        return parsed_contents

    def get_block_paths(self, blocknum):
        '''
        '''
        block_paths = self.log_load_section('block')
        for key, value in block_paths.items():
            try:
                block_paths[key] = value.format(blocknum)
            except IndexError:
                # pattern `{}` appears more than once
                ninstances = len(value.split('{}'))-1
                block_paths[key] = value.format(*[blocknum]*ninstances)
            except KeyError:
                # leave named place holders alone
                continue

        return block_paths

    def ephys_get_probe_hemisphere(self, hemisphere):
        '''
        '''
        probe_names = self.log_load_section('ephys', 'probes')
        for probe_name in probe_names:
            hemi = self.log_load_section(probe_name, 'nickname')
            if hemi == hemisphere:
                return probe_name

    def ephys_get_probe_object(self, probe_name=None, path=None):
        '''
        '''
        # log file contains a section
        # [probe_name]
        # phy_path = full/path/to/thing
        if path is None:
            probe_paths = self.log_load_section(probe_name)
            path = misc.uri_convert_uri2local(probe_paths['path_phy'])
        return phy.ProbeHandler(path)

    def ephys_get_probe_nclusters(self, probe_name):
        '''
        '''
        probe_object = self.ephys_get_probe_object(probe_name)
        probe_object.npy_template_feature_ind()
        nclusters = probe_object.npy_template_feature_ind.data.shape[0]
        return nclusters

    def ephys_get_recording_object(self, path=None):
        '''
        '''
        if path is None:
            # log file contains a section
            # [probe_name]
            # phy_path = full/path/to/thing
            ephys_info = self.log_load_section('ephys')
            ephys_path = ephys_info['path']
            spikeglx_prefix = self.log_load_section(
                'spikeglx')['recording_prefix']
            recording_path = os.path.join(ephys_path, spikeglx_prefix)
            path = misc.uri_convert_uri2local(recording_path)
        return phy.RecordingHandler(path)

    def ephys_get_pxinidaq_data(self, path=None,
                                index_npsync=0,
                                index_acqlive=1,
                                index_flipper=2):
        '''Load the PXI NI DAQ data as saved by SpikeGLX.
        '''
        # log file contains a section
        # [pxinidaq]
        # path = full/path/to/thing
        if path is None:
            paths = self.log_load_section('pxinidaq')
            path = misc.uri_convert_uri2local(paths['path'])

        pxinidaq_flname = path
        print('Loading: %s' % pxinidaq_flname)
        pxinidaq_metadata = spikeglx.read_metadata(pxinidaq_flname)
        pxinidaq_sample_ratehz = pxinidaq_metadata['niSampRate']
        pxinidaq_data = spikeglx.load_data_chunk(
            pxinidaq_flname, sample_duration=-1).T
        pxinidaq_time = np.arange(
            pxinidaq_data.shape[0])/pxinidaq_sample_ratehz

        pxinidaq = misc.DotDict(npsync=sync.analog2digital(pxinidaq_data[:, index_npsync]),   # NP digital signal
                                # acquisition live timeline signal)
                                acqlive=sync.analog2digital(
                                    pxinidaq_data[:, index_acqlive]),
                                flipper=sync.analog2digital(
                                    pxinidaq_data[:, index_flipper]),  # arduino flipper signal
                                times=pxinidaq_time,
                                sample_ratehz=pxinidaq_sample_ratehz)
        return pxinidaq

    def fusi_get_probe_localizer(self, probe_name, slice_name='slice00', is_ystack=False):
        '''
        '''
        section_header = 'location_%s_%s' % (slice_name, probe_name)
        paths = self.log_load_section(section_header)
        pprint(paths)
        from fusi import handler
        path = misc.uri_convert_uri2local(paths['fusi_data'])
        data = handler.matlab_data(path)
        if is_ystack is False:
            # Make sure log data matches MATLAB content
            assert np.allclose(data['yCoords'], paths['ycoord'])
            moving_probe = data['Doppler'].yStack.copy()
            del(data)
        else:
            # It's a ystack 4D image: (z,x,t,y)
            # default to using the first slice
            moving_probe = data['Doppler'].yStack[..., 0]
        return moving_probe

    def fusi_get_probe_slice_mask(self,
                                  probe_name,
                                  slice_name='slice00',
                                  roi_file_name='fusi_probe_roi.hdf',
                                  verbose=False):
        '''
        '''
        section_header = 'location_%s_%s' % (slice_name, probe_name)
        localizer_info = self.log_load_section(section_header)
        if verbose > 1:
            pprint(localizer_info)
        localizer_subfolder = str(localizer_info['folder_name'])
        flname = self.cortexlab_mk_filename(roi_file_name,
                                            subfolder=localizer_subfolder)
        mask = hdf_load(str(flname), 'mask', verbose=verbose > 1)
        if verbose:
            print('fUSi probe ROI voxels: %i' % mask.sum())
        return mask

    def fusi_get_probe_master_mask(self,
                                   probe_name,
                                   roi_file_name='fusi_probe_roi.hdf'):
        '''
        probe2Dprojection : 'fusi_probe_roi.hdf'
        V1 chunk : 'fusi_V1_section_in_probe_roi.hdf'
        HPC chunk : fusi_hippo_section_in_probe_roi.hdf'
        '''
        blocks = self.log_load_section('experiment', 'block_numbers')
        fusi_slices = self.log_load_section('fusi', 'mapping_block2slices')
        slices2blocks = {k: v for k, v in zip(fusi_slices, blocks)}
        fusi_slices = [t for t in np.unique(fusi_slices) if not np.isnan(t)]
        masks = np.asarray([self.fusi_get_probe_slice_mask(
            probe_name, 'slice%02i' % sdx, roi_file_name=roi_file_name).astype(np.int) for sdx in fusi_slices])
        master_mask = masks.sum(0) > 0
        return master_mask

    def fusi_get_probe_mask_extreme_coords(self, probe_name, slice_number=0, in_xyz_mm=False):
        '''Get top and bottom voxel positions

        (0,0,0) is left,back,top.

        Returns
        -------
        top_xyz, bottom_xyz
            x:ML: from the left
            y:AP: from the back
            z:DV: from the top
        '''
        mask = self.fusi_get_probe_slice_mask(
            probe_name, slice_name='slice%02i' % slice_number)
        # exclude outside of brain voxels
        slice_blocks = self.fusi_get_slice_blocks(slice_number)
        block = MetaBlock(self.subject_name, self.session_name,
                          slice_blocks[0], verbose=False)
        outside_brain = block.fusi_get_outsidebrain_mask()
        mask[outside_brain] = False

        dv_voxels = mask.sum(-1).nonzero()[0]
        ml_voxels = mask.sum(0).nonzero()[0]
        dv_top, dv_bottom = dv_voxels[0], dv_voxels[-1]
        ml_top = round(np.mean(mask[dv_top].nonzero()[0]))
        ml_bottom = round(np.mean(mask[dv_bottom].nonzero()[0]))
        if in_xyz_mm:
            ml_mm, dv_mm = self.fusi_image_pixel_mm
            ap_mm = self.fusi_get_slices_mm(slice_number)
            xyz = np.asarray([[ml_top, ml_bottom],
                              [ap_mm, ap_mm],
                              [dv_top, dv_bottom],
                              ])
            xyz[0, :] *= ml_mm
            xyz[2, :] *= dv_mm
            return xyz[:, 0], xyz[:, 1]

        # get position of slice in AP
        info = self.log_load_section('fusi')
        slices_mm = info['slice_positions']
        ystack_mm = info['ystack_positions']
        ap_pos = np.asarray([idx for idx, t in enumerate(
            ystack_mm) if t in slices_mm])[slice_number]
        return (ml_top, ap_pos, dv_top), (ml_bottom, ap_pos, dv_bottom)

    @property
    def fusi_shape(self):
        horizontal, vertical = self.fusi_get_coords_inplane_mm()
        return len(vertical), len(horizontal)

    def fusi_get_ystack(self):
        '''y-stack for session

        Raw data is stored as a 4D array: (nz, nx, nt, ny).
        This computes the mean across time

        Returns
        -------
        ystack : np.ndarray, (nz, nx, ny)
        '''
        from fusi import handler
        paths = self.log_load_section('fusi')
        path = misc.uri_convert_uri2local(paths['ystack_volume'])
        data = handler.matlab_data(path)
        out = data['Doppler'].yStack.mean(2)
        del(data)
        return out

    def fusi_get_ystack_nifti(self, nonlinearity=lambda x: x):
        '''Convert ystack to RAS-oriented nifti

        Applies sqrt non-linearity to data

        Returns
        -------
        image : Nifti1Image
        '''
        from fusi import align, handler

        # get volume
        arr = self.fusi_get_ystack()
        arr = nonlinearity(arr)

        # get dimensions
        xmm, zmm = self.fusi_image_pixel_mm

        paths = self.log_load_section('fusi')
        path = misc.uri_convert_uri2local(paths['ystack_volume'])
        data = handler.matlab_data(path)
        ymm = np.median(np.diff(data['yCoords']))
        xyzmm = np.asarray([xmm, ymm, zmm])
        print(arr.shape, xyzmm)

        im = align.fusiarr2nii(arr, xyzmm=xyzmm,
                               flips=(1, 1, -1))
        return im

    def fusi_get_allenccf_byindex(self, scale_factor=1):
        '''
        '''
        import nibabel as nib

        fl_resampled_atlas = self.cortexlab_mk_filename('alleccf_atlas_resampled_fusi_scaled%02ix_byindex.nii.gz' % scale_factor,
                                                        'allenccf_align')
        im = nib.load(fl_resampled_atlas)
        return np.asarray(im.get_fdata())

    def fusi_get_slices_mm(self, slice_number=None):
        '''
        '''
        info = self.log_load_section('fusi')
        slices_mm = info['slice_positions']
        if slice_number is not None:
            slices_mm = slices_mm[slice_number]
        return slices_mm

    def fusi_get_mean_slices(self, slice_number=None, scale_factor=1, remove_outside=True):
        '''
        '''
        nii = self.fusi_get_ystack_nifti()
        arr = np.asarray(nii.get_fdata())
        info = self.log_load_section('fusi')

        slices_mm = info['slice_positions']
        ystack_mm = info['ystack_positions']
        ystack_idx = np.asarray(
            [idx for idx, t in enumerate(ystack_mm) if t in slices_mm])

        ystack_nslices = len(ystack_mm)
        axis = np.asarray([idx == ystack_nslices for idx in arr.shape])
        assert axis.sum() == 1
        axis = int(axis.nonzero()[0])

        slicer = [slice(None)]*arr.ndim
        slicer[axis] = ystack_idx

        # Get relevant coronal slices and put them in the first dimension
        dat = arr[tuple(slicer)].transpose((1, 2, 0)).astype(np.int)

        if remove_outside:
            for slicenum in range(self.fusi_nslices):
                slice_blocks = self.fusi_get_slice_blocks(slicenum)
                blockob = MetaBlock(
                    self.subject_name, self.session_name, slice_blocks[0], verbose=False)
                outside_brain = blockob.fusi_get_outsidebrain_mask()
                dat[:, outside_brain] = 0

        if slice_number is not None:
            dat = dat[slice_number]

        return dat

    def fusi_get_allenccf_slices(self, slice_number, scale_factor=1, remove_outside=True):
        '''
        '''
        arr = self.fusi_get_allenccf_byindex(scale_factor)
        info = self.log_load_section('fusi')

        slices_mm = info['slice_positions']
        ystack_mm = info['ystack_positions']
        ystack_idx = np.asarray(
            [idx for idx, t in enumerate(ystack_mm) if t in slices_mm])

        ystack_nslices = len(ystack_mm)
        axis = np.asarray([idx == ystack_nslices for idx in arr.shape])
        assert axis.sum() == 1
        axis = int(axis.nonzero()[0])

        slicer = [slice(None)]*arr.ndim
        slicer[axis] = ystack_idx

        # Get relevant coronal slices and put them in the first dimension
        dat = arr[tuple(slicer)].transpose((1, 2, 0)).astype(np.int)

        if remove_outside:
            for slicenum in range(self.fusi_nslices):
                slice_blocks = self.fusi_get_slice_blocks(slicenum)
                blockob = MetaBlock(
                    self.subject_name, self.session_name, slice_blocks[0], verbose=False)
                outside_brain = blockob.fusi_get_outsidebrain_mask()
                dat[:, outside_brain] = 0

        if slice_number is not None:
            dat = dat[slice_number]

        return dat

    def fusi_show_allenccf_slices(self,
                                  slice_number=None,
                                  scale_factor=1,
                                  remove_outside=True,
                                  ax=None,
                                  alpha=0.5):
        '''
        '''
        dat = self.fusi_get_allenccf_slices(slice_number=slice_number,
                                            scale_factor=scale_factor,
                                            remove_outside=remove_outside)

        if ax is None:
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots()

        from fusi.align import allenccf_cmap
        cmap, norm = allenccf_cmap()
        im = ax.matshow(dat, cmap=cmap, norm=norm,
                        aspect=self.fusi_aspect_ratio, alpha=alpha)
        return ax, im

    def fusi_show(self, array_data, slice_number=0, ax=None,
                  allenccf_background=True, allenccf=True, allenccf_alpha=0.5, **kwargs):
        '''
        '''
        if array_data.ndim == 1:
            array_data = array_data.reshape(self.fusi_shape)

        if ax is None:
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots()

        if allenccf_background:
            im_allenccf = self.fusi_show_allenccf_slices(
                slice_number, ax=ax, alpha=allenccf_alpha)
            im = ax.matshow(
                array_data, aspect=self.fusi_aspect_ratio, **kwargs)
        else:
            im = ax.matshow(
                array_data, aspect=self.fusi_aspect_ratio, **kwargs)
            im_allenccf = self.fusi_show_allenccf_slices(
                slice_number, ax=ax, alpha=allenccf_alpha)
        return ax, (im, im_allenccf)

    def fusi_get_allenccf_contours(self, slice_number=0, area_indexes=None, minlen=50,
                                   ax=None, color=None):
        '''
        '''
        from skimage import measure

        allenccf_slice = self.fusi_get_allenccf_slices(slice_number)

        if area_indexes is None:
            from fusi import align
            area_indexes = align.allenccf_main_areas()

        contours = {areaidx: measure.find_contours(
            allenccf_slice, areaidx) for areaidx, color in area_indexes.items()}
        contours = {areaidx: [tt for tt in t if len(
            tt) > minlen] for areaidx, t in contours.items()}

        if ax is not None:
            for areaidx, area_contours in sorted(contours.items())[::-1]:
                for contour in area_contours:
                    ax.plot(contour[:, 1].astype(np.int),
                            contour[:, 0].astype(np.int), linewidth=1,
                            color='#%s' % area_indexes[areaidx] if color is None else color)
        return contours

    def fusi_get_probedepth_volume(self, probe_name, estimate_type='manual'):
        '''
        '''
        import nibabel as nib
        # load volume
        volfl = self.cortexlab_mk_filename('estimated_%s_3Dtrack_%s.nii.gz' % (probe_name, estimate_type),
                                           'allenccf_align')
        assert pathlib.Path(volfl).exists()
        vol = nib.load(volfl)
        return vol

    def fusi_get_probedepth_from_volume(self, probe_name, estimate_type='manual'):
        '''
        '''
        fl = self.cortexlab_mk_filename('estimated_%s_3Dtrack_%s.hdf' % (probe_name, estimate_type),
                                        'allenccf_align')
        xyzmm = readers.hdf_load(fl, 'probe_tip_mm', verbose=self.verbose)
        probe_depth = np.sqrt(np.sum(xyzmm**2))
        return probe_depth

    def fusi_get_slice_probedepth_and_voxels(self,
                                             slice_number,
                                             probe_name,
                                             widthmm=0.5,
                                             estimate_type='manual',
                                             remove_outside=True):
        '''
        '''
        probe_mask = self.fusi_get_probe_master_mask(probe_name)

        vol = self.fusi_get_probedepth_volume(
            probe_name, estimate_type=estimate_type)
        arr = np.asarray(vol.get_fdata())
        volxyzmm = vol.affine[np.diag_indices_from(np.eye(3))]
        del(vol)

        info = self.log_load_section('fusi')
        slices_mm = np.asarray(info['slice_positions'])
        ystack_mm = np.asarray(info['ystack_positions'])
        ystack_idx = np.asarray(
            [idx for idx, t in enumerate(ystack_mm) if t in slices_mm])
        ystack_nslices = len(ystack_mm)

        # determine volume axis for slices
        axis = np.asarray([idx == ystack_nslices for idx in arr.shape])
        assert axis.sum() == 1
        axis = int(axis.nonzero()[0])

        axis_nvox = arr.shape[axis]
        axis_mm = axis_nvox*volxyzmm[axis]

        slice_position = slices_mm[slice_number]
        slice_window_marker = np.logical_and(ystack_mm <= slice_position + widthmm/2.0,
                                             ystack_mm >= slice_position - widthmm/2.0)

        slicer = [slice(None)]*arr.ndim
        slicer[axis] = slice_window_marker.nonzero()[0]

        # Get relevant slices and put them in the first dimension
        dat = arr[tuple(slicer)]
        # remove probe section outside of penetration (marked with negative values)
        dat = np.where(dat < 0, np.nan, dat)
        # min so that we're biased towards the back of brain
        dat = np.nanmin(dat, axis=axis).T
        # also, force minimum value to be 0. it's not 0 b/c of voxelization
        assert np.nanmin(dat) >= 0
        if np.nanmin(dat) < 0.1:  # [mm]
            # if it's less than 100[um] set to zero
            dat[np.unravel_index(np.nanargmin(dat), dat.shape)] = 0

        # DV min/max
        probe_location = np.abs(np.nanmax(dat, axis=1)) > 0
        probe_mask[np.logical_not(probe_location), :] = 0

        if remove_outside:
            slice_blocks = self.fusi_get_slice_blocks(slice_number)
            blockob = MetaBlock(
                self.subject_name, self.session_name, slice_blocks[0], verbose=False)
            outside_brain = blockob.fusi_get_outsidebrain_mask()
            dat[outside_brain] = np.nan
            probe_mask[outside_brain] = 0

        npx_depth = self.fusi_get_probedepth_from_volume(probe_name)
        probe_depth = dat[~np.isnan(dat)]
        probe_bottom, probe_top = probe_depth.max(), probe_depth.min()  # 0:top of brain
        phy_limits = np.asarray(
            [npx_depth - probe_bottom, npx_depth - probe_top])
        if self.verbose:
            print('slicepos=%0.02f[mm] and slices:' %
                  slice_position, ystack_mm[slice_window_marker])
            print('probe depth:', phy_limits)
        return dat, probe_mask, phy_limits

    def fusi_get_coords_inplane_mm(self):
        '''Read image coordinates from fUSi acquistion

        Uses the size of a y-stack slice for all images.

        Returns
        -------
        horiz_mm_coords, vert_mm_coords
        '''
        from fusi import handler
        paths = self.log_load_section('fusi')
        path = misc.uri_convert_uri2local(paths['ystack_volume'])
        data = handler.matlab_data(path)
        hcoords = data['Doppler'].xAxis.copy()
        vcoords = data['Doppler'].zAxis.copy()
        del(data)
        return hcoords, vcoords

    @property
    def fusi_aspect_ratio(self):
        '''Assumes all images in this session have the same pixel resolution.

        Returns
        -------
        aspect_ratio = vert_mm/horiz_mm
        '''
        hsize, vsize = self.fusi_image_pixel_mm
        return vsize/hsize

    @property
    def fusi_image_pixel_mm(self):
        '''
        Returns
        -------
        horiz_size_mm, vert_size_mm

        '''
        hcoords, vcoords = self.fusi_get_coords_inplane_mm()
        hsize = np.median(np.unique(np.diff(hcoords)))
        vsize = np.median(np.unique(np.diff(vcoords)))
        return hsize, vsize

    @property
    def analysis_blocks(self):
        '''Blocks for analysis from the session LOG.
        '''
        return self.log_load_section('experiment', 'analysis_blocks')

    def ephys_get_cluster_depths(self, probe_name):
        '''
        '''
        nclusters = self.ephys_get_probe_nclusters(probe_name)
        depths = np.zeros(nclusters)*np.nan

        probe_object = self.ephys_get_probe_object(probe_name)
        # load cluster depths
        probe_object.tsv_cluster_info()
        # depth_um = probe_object.tsv_cluster_info.depth.astype(np.float32)

        if hasattr(probe_object.tsv_cluster_info, 'cluster_id'):
            # Latest PHY changed the heade
            depths[probe_object.tsv_cluster_info.cluster_id] = probe_object.tsv_cluster_info.depth.copy()
        else:
            depths[probe_object.tsv_cluster_info.id] = probe_object.tsv_cluster_info.depth.copy()
        return depths

    def ephys_get_mua_masks(self, probe_name, mua_window_um=500, neuropix_size_um=3840):
        '''
        '''
        mua_depths = np.arange(0, neuropix_size_um, mua_window_um)
        print(mua_depths)
        mua_nchunks = len(mua_depths)

        nclusters = self.ephys_get_probe_nclusters(probe_name)
        mask = np.zeros((mua_nchunks, nclusters)).astype(np.bool)

        probe_object = self.ephys_get_probe_object(probe_name)
        # load cluster depths
        probe_object.tsv_cluster_info()
        depth_um = probe_object.tsv_cluster_info.depth.astype(np.float32)
        mua_masks = futils.fast_find_between(
            depth_um, mua_depths, mua_window_um)
        for muaidx, mua_mask in enumerate(mua_masks):
            mask[muaidx, mua_mask] = True
        return mask

    def ephys_get_clusters_group(self, probe_name):
        '''
        '''
        nclusters = self.ephys_get_probe_nclusters(probe_name)
        probe_object = self.ephys_get_probe_object(probe_name)

        goods = np.zeros(nclusters).astype(np.bool)
        # get good clusters
        try:
            probe_object.tsv_cluster_group()  # load
            grp = probe_object.tsv_cluster_group.group
            cid = probe_object.tsv_cluster_group.cluster_id
        except:
            print('SPIKE SORTING NOT READY. USING AUTOMATED LABELS')
            probe_object.tsv_cluster_info()
            grp = probe_object.tsv_cluster_info.group
            cid = probe_object.tsv_cluster_info.id

        return cid, grp

    def ephys_get_putative_class(self, probe_name, neuron_class):
        '''
        '''
        assert neuron_class in ['excitatory', 'inhibitory']

        from fusi.io import phy
        probe_object = self.ephys_get_probe_object(probe_name)
        fl = probe_object.__pathobj__/'cluster_putative_neuronclass.csv'

        data = {k: v for k, v in phy.yield_table_columns(fl, delimiter=',')}

        nclusters = self.ephys_get_probe_nclusters(probe_name)
        goods = np.zeros(nclusters).astype(np.bool)

        class_marker = data['putative_class'] == neuron_class

        goods[data['cluster_id']] = class_marker
        return goods

    def ephys_get_good_clusters(self, probe_name):
        '''
        '''
        nclusters = self.ephys_get_probe_nclusters(probe_name)
        probe_object = self.ephys_get_probe_object(probe_name)

        goods = np.zeros(nclusters).astype(np.bool)
        # get good clusters
        try:
            probe_object.tsv_cluster_group()  # load
            mask = probe_object.tsv_cluster_group.group != 'noise'
            good_clusters = probe_object.tsv_cluster_group.cluster_id[mask]
        except:
            print('SPIKE SORTING NOT READY. USING AUTOMATED LABELS')
            probe_object.tsv_cluster_info()
            mask = probe_object.tsv_cluster_info.group != 'noise'
            good_clusters = probe_object.tsv_cluster_info.id[mask]

        goods[good_clusters] = True
        return goods

    def get_block(self, block_number=2):
        '''
        '''
        assert block_number in self.experiment_blocks
        return MetaBlock(self.subject_name, self.session_name, block_number)

    def generate_fusi_task_blocks(self, task_name):
        '''
        '''
        assert task_name in self.experiment_tasks
        task_blocks = self.experiment_get_task_blocks(task_name)
        for block in task_blocks:
            yield self.get_block(block_number=block)

    def generate_analysis_blocks(self):
        '''
        '''
        for block_number in self.analysis_blocks:
            block = MetaBlock(self.subject_name, self.session_name, block_number, verbose=self.verbose)
            if block.task_name in ('spontaneous', 'checkerboard'):
                yield block

    def generate_fusi_slice_blocks(self, slice_number=0):
        '''
        '''
        slice_blocks = self.fusi_get_slice_blocks(slice_number)
        print('Found %i blocks for slice=%i' %
              (len(slice_blocks), slice_number), slice_blocks)
        for block in slice_blocks:
            yield MetaBlock(self.subject_name, self.session_name, block)

    def generate_fusi_slice_blocks_method(self, slice_number,
                                          method_name,
                                          *args,
                                          **kwargs):
        '''
        Example
        -------
        >>> # Get all the data from the slices
        >>> fusi_iterator = subject.generate_fusi_slice_blocks_method(0, 'fusi_get_data')
        >>> times, data = next(method_iterator)
        >>> ephys_iterator = sub.generate_fusi_slice_blocks_method(0, 'ephys_get_probe_spikes_object', 'probe00')
        >>> probe = next(ephys_iterator)
        >>> nmua, mua_matrix = probe.time_locked_mua_matrix(times, dt=0.3)
        '''
        slice_blocks = self.generate_fusi_slice_blocks(slice_number)
        for block in slice_blocks:
            method = getattr(block, method_name)
            yield method(*args, **kwargs)

    def generate_slice_fusi_and_ephys_data(self, slice_number, probe_name, **fusi_kwargs):
        '''

        Example
        -------
        >>> iterator = sub.generate_slice_fusi_and_ephys_data(0, 'probe00')
        >>> times, fusi_data, nmua, mua_matrix = next(iterator)

        Yields
        ------
        fusi_times : 1D np.ndarray, (t,)
        fusi_data : 3D np.ndarray, (t, nz, nx)
        nmua : 1D np.ndarray, (m,)
        mua_matrix : 2D np.ndarray, (t, m)
        '''
        fusi_iterator = self.generate_fusi_slice_blocks_method(
            slice_number, 'fusi_get_data', **fusi_kwargs)
        ephys_iterator = self.generate_fusi_slice_blocks_method(
            slice_number, 'ephys_get_probe_spikes_object', probe_name)
        fusi_dt = fusi_kwargs.get('dt_ms', 300)/1000.

        for (times, fusi_data), probe_object in zip(fusi_iterator, ephys_iterator):
            nmua, mua_matrix = probe_object.time_locked_mua_matrix(
                times, dt=fusi_dt)
            yield times, fusi_data, nmua, mua_matrix

    def fusi_get_slice_blocks(self, slice_number=0):
        '''
        '''
        fusi_block_slices = self.log_load_section(
            'fusi', 'mapping_block2slices')
        blocks = self.log_load_section('experiment', 'block_numbers')
        slice_blocks = blocks[fusi_block_slices == slice_number]
        return slice_blocks

    def fusi_load_all_slice_data(self, slice_number,
                                 dt_ms=300,
                                 window_ms=400,
                                 svddrop=15,
                                 roi_name=None,
                                 concatenate=True):
        '''
        Parameters
        ----------
        slice_number (int): Index of the slice we want

        '''
        fusi_block_slices = self.log_load_section(
            'fusi', 'mapping_block2slices')
        blocks = self.log_load_section('experiment', 'block_numbers')
        slice_blocks = blocks[fusi_block_slices == slice_number]

        print(slice_blocks, slice_number, fusi_block_slices)
        slice_times = []
        slice_data = []
        for bdx, block_number in enumerate(slice_blocks):
            assert np.allclose(block_number, int(block_number))
            block_number = int(block_number)
            subject_block = MetaBlock(self.subject_name,
                                      self.session_name,
                                      block_number,
                                      root=self.root.parent)
            print(subject_block)

            # Load fUSi data
            times, data = subject_block.fusi_get_data(dt_ms=dt_ms,
                                                      window_ms=window_ms,
                                                      svddrop=svddrop)

            if bdx > 0 and concatenate:
                # changes times to be continuous with last block
                last_times = slice_times[bdx-1][-2:]
                times += last_times[-1] + np.diff(last_times) - times.min()

            if roi_name is not None:
                if 'probe' in roi_name:
                    mask = subject_block.fusi_get_probe_slice_mask(roi_name)
                    data = data[..., mask.astype(np.bool)]
                else:
                    raise ValueError('Unknown fUSi mask: %s' % roi_name)
            slice_times.append(times)
            slice_data.append(data)
        if concatenate:
            slice_times = np.hstack(slice_times)
            slice_data = np.vstack(slice_data)
        return slice_times, slice_data

    def load_physio_data(self,
                         slice_number,
                         dt_ms=300,
                         window_ms=400,
                         svddrop=15,
                         trim_beg=None,
                         trim_end=None,
                         concatenate=True):
        '''
        Parameters
        ----------
        slice_number (scalar) : slice code
        dt_ms (scalar)         : new sampling rate in [milliseconds]
        '''

        from fusi import resampling

        fusi_block_slices = self.log_load_section(
            'fusi', 'mapping_block2slices')
        blocks = self.log_load_section('experiment', 'block_numbers')
        slice_blocks = blocks[fusi_block_slices == slice_number]

        print(slice_blocks, slice_number, fusi_block_slices)
        timestamps = []
        slice_physio_data = []
        for bdx, block_number in enumerate(slice_blocks):
            assert np.allclose(block_number, int(block_number))
            block_number = int(block_number)
            subject_block = MetaBlock(self.subject_name,
                                      self.session_name,
                                      block_number,
                                      root=self.root.parent)
            print(subject_block)

            # Load fUSi data
            fusi_times, _ = subject_block.fusi_get_data(dt_ms=dt_ms,
                                                        window_ms=window_ms,
                                                        svddrop=svddrop)
            physio_times, physio_data = subject_block.fusi_get_heartbeat_estimate()

            # resample physio data to fUSi frequency
            physio_data = resampling.lanczos_resample(dt_ms/1000.,
                                                      physio_data[..., None],
                                                      physio_times,
                                                      fusi_times).squeeze()

            if bdx > 0 and concatenate:
                # changes times to be continuous with last block
                last_times = timestamps[bdx-1][-2:]
                fusi_times += last_times[-1] + \
                    np.diff(last_times) - fusi_times.min()

            # trim individual blocks
            if trim_beg:
                fusi_times = fusi_times[trim_beg:]
                physio_data = physio_data[trim_beg:]
            if trim_end:
                fusi_times = fusi_times[:trim_end]
                physio_data = physio_data[:trim_end]

            timestamps.append(fusi_times)
            slice_physio_data.append(physio_data)

        if concatenate:
            timestamps = np.hstack(timestamps)
            slice_physio_data = np.hstack(slice_physio_data)
        return timestamps, slice_physio_data

    def load_joint_data(self,
                        slice_number,
                        probe_name,
                        dt_ms=300,
                        window_ms=400,
                        svddrop=15,
                        freq_cutoffhz=15,
                        roi_name=None,
                        trim_beg=None,
                        trim_end=None,
                        recache=False,
                        mirrored=False,
                        concatenate=True):
        '''
        Parameters
        ----------
        slice_number (int): Index of the slice we want
        probe_name (str)
        trim_beg: data[beg:]
        trim_end: data[:end]

        Returns
        -------
        times
        fusi_data
        spike_data
        '''
        params = OrderedDict(locals())
        params['path'] = self.session_path
        params['name'] = 'joint_data'
        cache_file = params2cache(params)
        print(cache_file)

        if cache_file.exists() and recache is False:
            timestamps = readers.hdf_load(cache_file, 'timestamps')
            slice_fusi_data = readers.hdf_load(cache_file, 'slice_fusi_data')
            spike_data = readers.hdf_load(cache_file, 'spike_data')
            return timestamps, slice_fusi_data, spike_data

        fusi_block_slices = self.log_load_section(
            'fusi', 'mapping_block2slices')
        blocks = self.log_load_section('experiment', 'block_numbers')
        slice_blocks = blocks[fusi_block_slices == slice_number]
        nclusters = self.ephys_get_probe_nclusters(probe_name)

        print(slice_blocks, slice_number, fusi_block_slices)
        timestamps = []
        slice_fusi_data = []
        spike_data = []
        for bdx, block_number in enumerate(slice_blocks):
            assert np.allclose(block_number, int(block_number))
            block_number = int(block_number)
            subject_block = MetaBlock(self.subject_name,
                                      self.session_name,
                                      block_number,
                                      root=self.root.parent)
            print(subject_block)

            # Load fUSi data
            if not np.isscalar(freq_cutoffhz) and roi_name is not None:
                assert len(freq_cutoffhz) == 2
                # Bandpassed data is stored for ROIs
                times, fusi_data = subject_block.fusi_get_data(dt_ms=dt_ms,
                                                               window_ms=window_ms,
                                                               svddrop=svddrop,
                                                               freq_cutoffhz=freq_cutoffhz,
                                                               roi_name=roi_name,
                                                               mirrored=mirrored)
            else:
                # WHOLE BRAIN
                times, fusi_data = subject_block.fusi_get_data(dt_ms=dt_ms,
                                                               window_ms=window_ms,
                                                               svddrop=svddrop,
                                                               freq_cutoffhz=freq_cutoffhz)

            spike_times, spike_clusters = subject_block.ephys_get_probe_data(
                probe_name)
            spike_matrix = futils.bin_spikes(times,
                                             dt_ms/1000.,
                                             spike_times,
                                             spike_clusters,
                                             nclusters=nclusters)

            if bdx > 0 and concatenate:
                # changes times to be continuous with last block
                last_times = timestamps[bdx-1][-2:]
                times += last_times[-1] + np.diff(last_times) - times.min()

            if roi_name is not None:
                if ('probe' in roi_name) and np.isscalar(freq_cutoffhz):
                    # only computed for these
                    # mask = subject_block.fusi_get_probe_slice_mask(roi_name) #
                    mask = subject_block.fusi_get_probe_master_mask(roi_name)
                    if mirrored:
                        # CONTROL: mirror and flip mask
                        mask = mask[::-1, ::-1]
                    fusi_data = fusi_data[..., mask.astype(np.bool)]
                elif (('probe' in roi_name) or ('outcenter' in roi_name)) and isinstance(freq_cutoffhz, (tuple, list)):
                    # data is already masked
                    pass
                else:
                    raise ValueError('Unknown fUSi mask: %s' % roi_name)

            # trim individual blocks
            if trim_beg:
                times = times[trim_beg:]
                fusi_data = fusi_data[trim_beg:]
                spike_matrix = spike_matrix[trim_beg:]
            if trim_end:
                times = times[:trim_end]
                fusi_data = fusi_data[:trim_end]
                spike_matrix = spike_matrix[:trim_end]

            timestamps.append(times)
            slice_fusi_data.append(fusi_data)
            spike_data.append(spike_matrix)

        if concatenate:
            timestamps = np.hstack(timestamps)
            spike_data = np.vstack(spike_data)
            slice_fusi_data = np.vstack(slice_fusi_data)

        readers.hdf_dump(cache_file, {'timestamps': timestamps,
                                      'slice_fusi_data': slice_fusi_data,
                                      'spike_data': spike_data})

        return timestamps, slice_fusi_data, spike_data

    def load_joint_spectrogram_data(self,
                                    slice_number,
                                    probe_name,
                                    dt_ms=300,
                                    window_ms=400,
                                    roi_name='probe00',
                                    trim_beg=None,
                                    trim_end=None,
                                    rawfft=False,
                                    rawpsd=False,
                                    concatenate=True,
                                    recache=False):
        '''
        Parameters
        ----------
        slice_number (int): Index of the slice we want
        probe_name (str)
        trim_beg: data[beg:]
        trim_end: data[:end]

        Returns
        -------
        times
        fusi_data
        spike_data
        '''
        params = OrderedDict(locals())
        params['path'] = self.session_path
        cache_file = params2cache(params)
        print(cache_file)

        if cache_file.exists() and recache is False:
            timestamps = readers.hdf_load(cache_file, 'timestamps')
            slice_fusi_data = readers.hdf_load(cache_file, 'slice_fusi_data')
            spike_data = readers.hdf_load(cache_file, 'spike_data')
            return timestamps, slice_fusi_data, spike_data

        fusi_block_slices = self.log_load_section(
            'fusi', 'mapping_block2slices')
        blocks = self.log_load_section('experiment', 'block_numbers')
        slice_blocks = blocks[fusi_block_slices == slice_number]
        nclusters = self.ephys_get_probe_nclusters(probe_name)

        print(slice_blocks, slice_number, fusi_block_slices)
        timestamps = []
        slice_fusi_data = []
        spike_data = []
        for bdx, block_number in enumerate(slice_blocks):
            assert np.allclose(block_number, int(block_number))
            block_number = int(block_number)
            subject_block = MetaBlock(self.subject_name,
                                      self.session_name,
                                      block_number,
                                      root=self.root.parent)
            print(subject_block)

            # Load fUSi data
            times, fusi_data = subject_block.fusi_get_roi_spectrogram(dt_ms=dt_ms,
                                                                      window_ms=window_ms,
                                                                      roi_name=roi_name,
                                                                      rawfft=rawfft,
                                                                      rawpsd=rawpsd,
                                                                      )

            spike_times, spike_clusters = subject_block.ephys_get_probe_data(
                probe_name)
            spike_matrix = futils.bin_spikes(times,
                                             dt_ms/1000.,
                                             spike_times,
                                             spike_clusters,
                                             nclusters=nclusters)

            if bdx > 0 and concatenate:
                # changes times to be continuous with last block
                last_times = timestamps[bdx-1][-2:]
                times += last_times[-1] + np.diff(last_times) - times.min()

            # trim individual blocks
            if trim_beg:
                times = times[trim_beg:]
                fusi_data = fusi_data[trim_beg:]
                spike_matrix = spike_matrix[trim_beg:]
            if trim_end:
                times = times[:trim_end]
                fusi_data = fusi_data[:trim_end]
                spike_matrix = spike_matrix[:trim_end]

            timestamps.append(times)
            slice_fusi_data.append(fusi_data)
            spike_data.append(spike_matrix)

        if concatenate:
            timestamps = np.hstack(timestamps)
            spike_data = np.vstack(spike_data)
            slice_fusi_data = np.vstack(slice_fusi_data)

        readers.hdf_dump(cache_file, {'timestamps': timestamps,
                                      'slice_fusi_data': slice_fusi_data,
                                      'spike_data': spike_data})

        return timestamps, slice_fusi_data, spike_data


class MetaBlock(MetaSession):
    '''
    '''

    def __init__(self,
                 subject_name,
                 session_name,
                 block_name,
                 root=None,
                 verbose=False):
        '''
        '''
        super(type(self), self).__init__(subject_name,
                                         session_name,
                                         root=root,
                                         verbose=verbose)

        block_name = str(block_name)
        block_number = int(block_name)
        block_path = self.session_path.joinpath(block_name)

        self.verbose = verbose
        self.block_name = block_name
        self.block_number = block_number
        self.block_path = block_path
        assert block_path.exists()

    def get_meta_session(self):
        return MetaSession(self.subject_name, self.session_name, verbose=False)

    @property
    def meta_session(self):
        return MetaSession(self.subject_name, self.session_name, verbose=False)

    @property
    def task_name(self):
        '''
        '''
        session_info = self.log_load_section('experiment')
        experiment_names = session_info['block_names']
        for expname in experiment_names:
            if int(self.block_number) in session_info[expname]:
                experiment_name = expname
                break
        return experiment_name.split('_')[1]

    @property
    def slice_number(self):
        '''
        '''
        slice_number = self.experiment_mapper_blocks2slices[self.block_number]
        return slice_number

    @property
    def analysis_slicewidth_search(self):
        search_widths = self.log_load_section('experiment', 'analysis_widthmm')
        return float(search_widths[self.analysis_blocks == self.block_number])

    def stimulus_load_times(self):
        timeline_aligned = self.timeline_get_aligned_times()
        # load timeline
        self.cortexlab_timeline_load_block()
        # get stimulus times
        self.timeline.load_photod_stimuli(timeline_aligned)
        stimulus_frame_times = self.timeline.phd_frame_times.copy()
        stimulus_start_times = self.timeline.phd_stim_start.copy()
        stimulus_end_times = self.timeline.phd_stim_end.copy()
        return stimulus_start_times, stimulus_end_times

    def stimulus_checkerboard_times(self):
        '''
        '''
        stimulus_start_times, stimulus_end_times = self.stimulus_load_times()
        nstimuli, nreps = stimulus_start_times.shape
        ordered_times = np.sort(stimulus_start_times.T.reshape(
            int(nreps/2), int(nstimuli*2)), 1).T

        self.timeline.load_protocol()
        print(self.timeline.protocol.pardefs)
        stimulus_parameters = self.timeline.protocol.pars
        xypos = stimulus_parameters[1:5].T     # (xleft, right, bottom, top)
        contrast = stimulus_parameters[[-1]].T  # contrast
        stimulus_params = np.hstack([xypos, contrast])
        checkerboard_params, trial_ids = np.unique(
            stimulus_params, axis=0, return_inverse=True)
        checkerboard_ids = np.unique(trial_ids)
        trial_names = {trialid: str(
            checkerboard_params[trialid]) for trialid in np.unique(trial_ids)}
        stimulus_matrix = np.asarray([trial_ids]*nreps).T
        stimulus_onsets = np.asarray([stimulus_start_times[stimulus_matrix == cid]
                                      for cid in checkerboard_ids]).T
        return stimulus_start_times, stimulus_matrix

    def stimulus_checkerboard_repeats(self):
        '''
        '''
        stimulus_start_times, stimulus_matrix = self.stimulus_checkerboard_times()
        repeat_ids = np.asarray([np.hstack([forward, backward]) for forward, backward in
                                 zip(stimulus_matrix[:, ::2].T, stimulus_matrix[:, 1::2][::-1].T)]).T
        # second half is repeated backwards
        assert np.allclose(repeat_ids[:20, 0], repeat_ids[20:, 0][::-1])
        repeat_times = np.asarray([np.hstack([forward, backward]) for forward, backward in
                                   zip(stimulus_start_times[:, ::2].T, stimulus_start_times[:, 1::2][::-1].T)]).T
        return repeat_times, repeat_ids

    def cortexlab_mk_filename(self, filename, subfolder=None, mkdir=False):
        date_txt = misc.date_tuple2cortexlab(self.date_tuple)
        flname = '{date}_{block}_{subject}_{fl}'.format(date=date_txt,
                                                        block=self.block_name,
                                                        subject=self.subject_name,
                                                        fl=filename)
        if subfolder is None:
            outpath = self.block_path
        else:
            outpath = self.block_path.joinpath(subfolder)
            if mkdir:
                outpath.mkdir(exist_ok=True)

        return outpath.joinpath(flname)

    def __str__(self):
        info = (self.subject_name, self.session_name, self.block_name)
        return 'subject=%s, session=%s, block=%s' % info

    def __repr__(self):
        info = (__name__, type(self).__name__,
                self.subject_name, self.session_name, self.block_name)
        return '<%s.%s [%s: session=%s, block=%s]>' % info

    @property
    def block_paths(self):
        return self.get_block_paths(self.block_name)

    def _get_hdf_field(self, filename, field):
        '''
        '''
        return readers.hdf_load(filename, field)

    def ephys_get_probe_spikes_object(self, probe_name, good_clusters=None):
        '''
        '''
        spike_times, spike_clusters = self.ephys_get_probe_data(probe_name)
        nclusters = self.ephys_get_probe_nclusters(probe_name)
        if good_clusters is None:
            good_clusters = self.ephys_get_good_clusters(probe_name)
        cluster_depths = self.ephys_get_cluster_depths(probe_name)
        from fusi.io import spikes
        return spikes.ProbeSpikes(spike_times, spike_clusters,
                                  nclusters=nclusters,
                                  good_clusters=good_clusters,
                                  cluster_depths=cluster_depths)

    def ephys_get_mua(self, probe_name, times, dt, **kwargs):
        '''
        '''
        probe_object = self.ephys_get_probe_spikes_object(probe_name)
        return probe_object.time_locked_mua_matrix(times, dt, **kwargs)

    def ephys_load_spikes(self, probe_name):
        '''
        '''
        filename = self.block_paths['%s_spikes' % probe_name]
        clusters = hdf_load(filename, 'spike_clusters')
        sample_ratehz = hdf_load(filename, 'phy_sample_ratehz')
        # spike times are stored in units of sample rate
        times = hdf_load(filename, 'spike_times')/sample_ratehz

        dotdict = misc.DotDict(clusters=clusters,
                               times=times,
                               sample_ratehz=sample_ratehz)

        setattr(self, probe_name, dotdict)

    def ephys_load_digital(self, probe_name):
        '''
        '''
        filename = self.block_paths['%s_' % probe_name]
        clusters = hdf_load(filename, 'spike_digital')
        sample_ratehz = hdf_load(filename, 'phy_sample_ratehz')
        # spike times are stored in units of sample rate
        times = hdf_load(filename, 'spike_times')/sample_ratehz

        dotdict = misc.DotDict(clusters=clusters,
                               times=times,
                               sample_ratehz=sample_ratehz)

        setattr(self, probe_name, dotdict)

    def cortexlab_timeline_load_block(self):
        '''
        '''
        timeline = cxlabexp.ExperimentData(self.subject_name,
                                           expnum=int(self.block_name),
                                           date=self.date_tuple,
                                           root=self.root.parent)
        setattr(self, 'timeline', timeline)

    def fusi_get_slice_probedepth_and_voxels(self,
                                             probe_name,
                                             widthmm=None,
                                             estimate_type='manual',
                                             remove_outside=True):
        '''
        '''
        if widthmm is None:
            widthmm = self.analysis_slicewidth_search
        slice_number = self.slice_number
        out = self.meta_session.fusi_get_slice_probedepth_and_voxels(slice_number,
                                                                     probe_name,
                                                                     widthmm=widthmm,
                                                                     estimate_type=estimate_type,
                                                                     remove_outside=remove_outside)
        return out

    def fusi_get_probe_allen_mask(self, probe_name, allen_area_name=None, fusi_mirror=False, criterion=1):
        '''
        '''
        from fusi import allen
        # OLD:'fusi_%s_section_in_probe_roi.hdf'%FUSI_PROBE_MASK_NAME)
        probe_mask = self.meta_session.fusi_get_probe_master_mask(probe_name)
        brain_mask = np.logical_not(self.fusi_get_outsidebrain_mask())
        pmask = np.logical_and(brain_mask, probe_mask)

        # ROI voxels in Allen Atlas
        allenccf_areas = self.meta_session.fusi_get_allenccf_slices(
            self.slice_number)
        area_mask = allen.mk_area_mask_from_aligned(
            allenccf_areas, allen_area_name, verbose=False)
        # Neuropix fUSI track within allen area ROI
        pmask = np.logical_and(area_mask, pmask)

        if fusi_mirror is True:
            # Mirror 2D projection for bilateral analysis
            # (This way b/c the probe track might not hit the area if mirrored first)
            pmask = pmask[:, ::-1]

        # 3D projection of NPx probe in fUSI
        # slice_position_mm = subject_block.meta_session.fusi_slice_position(subject_block.slice_number)
        slice_search_widthmm = self.analysis_slicewidth_search

        fusi_probe_depth, fusi_probe_mask, probe_limits_mm = self.meta_session.fusi_get_slice_probedepth_and_voxels(
            self.slice_number, probe_name, widthmm=slice_search_widthmm)

        # Percentage of 3D NPx probe mask inside the Allen Mask
        pct_probe_in_roi = (np.logical_and(
            fusi_probe_mask, pmask).sum()/fusi_probe_mask.sum())*100
        return None if pct_probe_in_roi < criterion else pmask

    def fusi_get_allen_area_mask(self,
                                 allen_area_name=None,
                                 fusi_mirror=False,
                                 hemisphere=None,
                                 exclude_probe=None):
        '''
        exclude_probe : str
            The probe name ROI to exclude (e.g. 'probe00')
            If given, the probe ROI is removed from the area ROI.
        '''
        from fusi import allen
        # OLD:'fusi_%s_section_in_probe_roi.hdf'%FUSI_PROBE_MASK_NAME)
        brain_mask = np.logical_not(self.fusi_get_outsidebrain_mask())

        # ROI voxels in Allen Atlas
        allenccf_areas = self.meta_session.fusi_get_allenccf_slices(
            self.slice_number)
        area_mask = allen.mk_area_mask_from_aligned(
            allenccf_areas, allen_area_name, verbose=False)
        # Neuropix fUSI track within allen area ROI
        allen_roi_mask = np.logical_and(area_mask, brain_mask)

        if fusi_mirror is True:
            # Mirror 2D projection for bilateral analysis
            # (This way b/c the probe track might not hit the area if mirrored first)
            allen_roi_mask = allen_roi_mask[:, ::-1]

        if hemisphere is not None:
            hemi_mask = self.fusi_get_hemisphere_mask(hemisphere)
            allen_roi_mask = np.logical_and(allen_roi_mask, hemi_mask)

        if exclude_probe is not None:
            probe_mask = self.fusi_get_probe_allen_mask(exclude_probe,
                                                        allen_area_name=allen_area_name,
                                                        fusi_mirror=fusi_mirror)
            allen_roi_mask = np.logical_and(
                allen_roi_mask, np.logical_not(probe_mask))

        return allen_roi_mask

    def fusi_get_outsidebrain_mask(self, verbose=False):
        '''
        '''
        fl = self.cortexlab_mk_filename(
            'fusi_outsidebrain_roi.hdf', subfolder='fusi')
        mask = hdf_load(str(fl), 'outsidebrain_mask', verbose=verbose)
        return mask.astype(np.bool)

    def fusi_get_hemisphere_mask(self, hemisphere, mask_inside_brain=True):
        '''Get a mask of the left or right hemisphere:
        hemisphere : str,
            One of 'LH' or 'RH'
        '''

        brain_mask = np.logical_not(self.fusi_get_outsidebrain_mask())
        midline_mask = self.fusi_get_midline_mask()
        horiz_midpoint = int(np.median(midline_mask.nonzero()[1]))
        hemisphere_mask = np.zeros_like(brain_mask)
        if hemisphere == 'RH':
            hemisphere_mask[:, horiz_midpoint:] = True
        else:
            hemisphere_mask[:, :horiz_midpoint] = True

        if mask_inside_brain:
            hemisphere_mask = np.logical_and(brain_mask, hemisphere_mask)
        return hemisphere_mask

    def fusi_get_midline_mask(self):
        '''
        '''
        fl = self.cortexlab_mk_filename(
            'fusi_midline_roi.hdf', subfolder='fusi')
        # name was saved incorrectly =S
        mask = hdf_load(str(fl), 'outsidebrain_mask')
        return mask.astype(np.bool)

    def fusi_get_raw_data_object(self, dataroot='.', nmax=1):
        '''
        Parameters
        ----------
        dataroot (str) : Path for DAT files
        nmax (int)     : Number of images to load
        '''
        from fusi.io import contacq
        dataroot = pathlib.Path(dataroot)
        data_path = dataroot.joinpath(self.subject_name,
                                      misc.date_tuple2cortexlab(
                                          self.date_tuple),
                                      self.block_name)
        fusi_loader = contacq.MetaLoader(
            data_path, verbose=True, nmaximum=nmax)
        return fusi_loader

    def fusi_get_heartbeat_estimate(self, key='filtered_vascular_timecourse_raw', clip=True):
        '''
        '''
        fl = self.block_path.joinpath('fusi', 'physio_hearbeat_estimate.hdf')
        physio = hdf_load(fl, key)

        fusi_loader = self.fusi_get_raw_data_object()
        START_DROPPED_FRAMES = 5
        # the first 5 "frames" (i.e. data bursts) have no TTL
        dropped_samples = START_DROPPED_FRAMES*fusi_loader.images_per_frame
        physio = physio[dropped_samples:]

        physio_dtms = 2  # 500[Hz]
        ttl_dtms = 100  # 10[Hz]

        # these are at a fixed 100ms
        fusi_ttl_onsets = self.fusi_get_ttl_times()

        from scipy import interpolate
        ttl_samples = np.arange(len(fusi_ttl_onsets))
        interpolator = interpolate.interp1d(
            ttl_samples, fusi_ttl_onsets, fill_value='extrapolate')
        # factor = dt_ms/ttl_dt
        factor = physio_dtms/ttl_dtms
        new_samples = np.arange(physio.shape[0])*factor
        physio_times = interpolator(new_samples)

        physio /= np.std(physio)
        if clip:
            physio = np.clip(physio, -5, 5)
        return physio_times, physio

    def fusi_get_probe_slice_mask(self, probe_name,
                                  slice_name=None,
                                  roi_file_name='fusi_probe_roi.hdf',
                                  verbose=False):
        '''
        '''
        fusi_block_slices = self.log_load_section(
            'fusi', 'mapping_block2slices')
        block_index = self.block_number - 1
        fusi_slice_number = fusi_block_slices[block_index]
        if slice_name is None:
            slice_name = 'slice%02i' % fusi_slice_number
        if verbose:
            print(slice_name, block_index)
        section_header = 'location_%s_%s' % (slice_name, probe_name)
        localizer_info = self.log_load_section(section_header)
        if verbose:
            pprint(localizer_info)

        localizer_subfolder = str(localizer_info['folder_name'])
        mask_flname = super(type(self), self).cortexlab_mk_filename(roi_file_name,
                                                                    subfolder=localizer_subfolder)

        # return mask_flname
        mask = hdf_load(str(mask_flname), 'mask', verbose=verbose)
        if verbose:
            print('fUSi probe ROI voxels: %i' % mask.sum())
        return mask

    def ephys_get_probe_data(self, probe_name, verbose=True):
        '''
        '''
        flname = self.cortexlab_mk_filename(
            '%s_spikes.hdf' % probe_name, subfolder='aligned2pxi')
        assert flname.exists()

        if verbose:
            print(f'Loading spike data: {pathlib.Path(flname).name}...')

        clusters = hdf_load(flname, 'clusters')
        times = hdf_load(flname, 'times')
        return times, clusters

    def timeline_get_aligned_times(self, key='times'):
        timeline_flname = self.cortexlab_mk_filename('timeline_aligned2pxi.hdf',
                                                     subfolder='aligned2pxi')
        timeline_newtimes = readers.hdf_load(timeline_flname, key)
        return timeline_newtimes

    def fusi_get_ttl_times(self):
        '''
        '''
        self.cortexlab_timeline_load_block()
        fusi_ttl_signal = self.timeline.get_timeline_data('neuralFrames')[1]
        fusi_ttl_signal = sync.remove_single_datapoint_onsets(fusi_ttl_signal)
        # load new timeline times (aligned to PXI NI DAQ)
        timeline_newtimes = self.timeline_get_aligned_times()
        # compute the onsets from the TTL using the new times
        ttl_dt = 100  # TTL is updated every 100[ms]
        fusi_ttl_onsets = sync.digital2times(
            timeline_newtimes, fusi_ttl_signal)[0]
        return fusi_ttl_onsets

    def fusi_get_times(self, **kwargs):
        '''Call to fusi_get_data that only returns the times
        '''
        return self.fusi_get_data(**kwargs)[0]

    def fusi_get_data(self, dt_ms=300, window_ms=400, svddrop=15,
                      freq_cutoffhz=15, roi_name=None, mirrored=False, verbose=True):
        '''Specify the parameters of the preprocessed fUSi data to load

        Parameters
        ----------
        dt_ms     : sample interval [in millisecond]
        window_ms : window over which filtering and svd drop is applied
        svddrop   : number of SVD components that are thrown on
        roi_name : name of ROI
        mirrored : control region

        Returns
        -------
        fusi_times : 1D np.ndarray (n,)
                data time-stamps aligned to PXI
        fusi_data  : 3D np.ndarray (n, vdim, hdim):
            fUSi data preprocessed with the speicified parameters
        '''
        # get time stamps
        ##############################
        # load the neural frames TTL signal from timeline
        self.cortexlab_timeline_load_block()
        fusi_ttl_signal = self.timeline.get_timeline_data('neuralFrames')[1]
        fusi_ttl_signal = sync.remove_single_datapoint_onsets(fusi_ttl_signal)
        # load new timeline times (aligned to PXI NI DAQ)
        timeline_newtimes = self.timeline_get_aligned_times()
        # compute the onsets from the TTL using the new times
        ttl_dt = 100  # TTL is updated every 100[ms]
        fusi_ttl_onsets = sync.digital2times(
            timeline_newtimes, fusi_ttl_signal)[0]

        # load the preprocessed fusi data
        ########################################
        if np.isscalar(freq_cutoffhz):
            assert freq_cutoffhz == 15  # UNDEFINED STORAGE FOR OTHERS

            if (svddrop != 5 and dt_ms == 50) or (svddrop != 15 and dt_ms == 300):
                pattern = '{yyyymmdd}_sess{blocknum}_{dt}ms_window{window}ms_svddrop{ndropped}_highpasscutoff%0.02fHz.hdf' % freq_cutoffhz
                flname = pattern.format(yyyymmdd=misc.date_tuple2number(self.date_tuple),
                                        blocknum='%02i' % self.block_number,
                                        dt='%03i' % dt_ms,
                                        window='%04i' % window_ms,
                                        ndropped=svddrop if svddrop is None else '%03i' % svddrop)
            else:
                pattern = '{yyyymmdd}_sess{blocknum}_{dt}ms_window{window}ms_svddrop{ndropped}.hdf'
                flname = pattern.format(yyyymmdd=misc.date_tuple2number(self.date_tuple),
                                        blocknum='%02i' % self.block_number,
                                        dt='%03i' % dt_ms,
                                        window='%04i' % window_ms,
                                        ndropped='%03i' % svddrop)

        elif isinstance(freq_cutoffhz, (tuple, list)):
            assert len(freq_cutoffhz) == 2
            low_cutoff, high_cutoff = freq_cutoffhz
            # # OLD
            # pattern = '{yyyymmdd}_sess{blocknum}_{dt}ms_window{window}ms_svddrop{ndropped}_bandpassL{low}hzH{high}hz.hdf'
            if roi_name is None:
                pattern = '{yyyymmdd}_sess{blocknum}_{dt}ms_window{window}ms_svddrop{ndropped}_bandpassed.hdf'
                flname = pattern.format(yyyymmdd=misc.date_tuple2number(self.date_tuple),
                                        blocknum='%02i' % self.block_number,
                                        dt='%03i' % dt_ms,
                                        window='%04i' % window_ms,
                                        ndropped=('%03i' % svddrop if isinstance(
                                            svddrop, int) else svddrop),
                                        # low='%03i'%low_cutoff,
                                        # high='%03i'%high_cutoff
                                        )

            else:
                # for probe00 and probe01
                if mirrored:
                    # control data: mirrored and flipped
                    pattern = '{yyyymmdd}_sess{blocknum}_{dt}ms_window{window}ms_svddrop{ndropped}_bandpassed_{roi_name}_mirrored.hdf'
                else:
                    # standard data
                    pattern = '{yyyymmdd}_sess{blocknum}_{dt}ms_window{window}ms_svddrop{ndropped}_bandpassed_{roi_name}.hdf'
                flname = pattern.format(yyyymmdd=misc.date_tuple2number(self.date_tuple),
                                        blocknum='%02i' % self.block_number,
                                        dt='%03i' % dt_ms,
                                        window='%04i' % window_ms,
                                        ndropped=('%03i' % svddrop if isinstance(
                                            svddrop, int) else svddrop),
                                        roi_name=roi_name)

        fusi_preproc_fl = self.block_path.joinpath('fusi', flname)
        if not fusi_preproc_fl.exists():
            raise IOError('Does not exist: %s' % fusi_preproc_fl)
        if verbose:
            print(f'Loading fUSI data: {pathlib.Path(fusi_preproc_fl).name}...')
        if np.isscalar(freq_cutoffhz):
            fusi_data = readers.hdf_load(fusi_preproc_fl, 'data')
        else:
            name = 'data_bandpass_%i_%i' % freq_cutoffhz
            fusi_data = readers.hdf_load(fusi_preproc_fl, name)
        if fusi_data.ndim == 4:
            fusi_data = fusi_data.mean(1)

        # get upsampled time stamps from the TTL
        from scipy import interpolate
        ttl_samples = np.arange(len(fusi_ttl_onsets))
        interpolator = interpolate.interp1d(
            ttl_samples, fusi_ttl_onsets, fill_value='extrapolate')
        factor = dt_ms/ttl_dt
        new_samples = np.arange(fusi_data.shape[0])*factor
        fusi_times = interpolator(new_samples)
        return fusi_times, fusi_data

    def fusi_get_roi_spectrogram(self, dt_ms=300, window_ms=400, roi_name='probe00',
                                 rawfft=False, rawpsd=False):
        '''Specify the parameters of the preprocessed fUSi data to load

        Parameters
        ----------
        dt_ms     : sample interval [in millisecond]
        window_ms : window over which filtering and svd drop is applied

        Returns
        -------
        fusi_times : 1D np.ndarray (n,)
                data time-stamps aligned to PXI
        fusi_data  : 3D np.ndarray (n, vdim, hdim):
            fUSi data preprocessed with the speicified parameters
        '''
        # get time stamps
        ##############################
        # load the neural frames TTL signal from timeline
        self.cortexlab_timeline_load_block()
        fusi_ttl_signal = self.timeline.get_timeline_data('neuralFrames')[1]
        fusi_ttl_signal = sync.remove_single_datapoint_onsets(fusi_ttl_signal)
        # load new timeline times (aligned to PXI NI DAQ)
        timeline_newtimes = self.timeline_get_aligned_times()
        # compute the onsets from the TTL using the new times
        ttl_dt = 100  # TTL is updated every 100[ms]
        fusi_ttl_onsets = sync.digital2times(
            timeline_newtimes, fusi_ttl_signal)[0]

        # load the preprocessed fusi data
        ########################################
        if rawfft:
            pattern = '{yyyymmdd}_sess{blocknum}_{dt}ms_window{window}ms_fUSiROI{roi_name}_rawfft.hdf'
        elif rawpsd:
            pattern = '{yyyymmdd}_sess{blocknum}_{dt}ms_window{window}ms_fUSiROI{roi_name}_rawpsd.hdf'
        else:
            pattern = '{yyyymmdd}_sess{blocknum}_{dt}ms_window{window}ms_fUSiROI{roi_name}_psd.hdf'

        flname = pattern.format(yyyymmdd=misc.date_tuple2number(self.date_tuple),
                                blocknum='%02i' % self.block_number,
                                dt='%03i' % dt_ms,
                                window='%04i' % window_ms,
                                roi_name=roi_name)

        fusi_preproc_fl = self.block_path.joinpath('fusi', flname)
        if not fusi_preproc_fl.exists():
            raise IOError('Does not exist: %s' % fusi_preproc_fl)
        print(fusi_preproc_fl)
        fusi_data = readers.hdf_load(fusi_preproc_fl, 'data')

        # get upsampled time stamps from the TTL
        from scipy import interpolate
        ttl_samples = np.arange(len(fusi_ttl_onsets))
        interpolator = interpolate.interp1d(
            ttl_samples, fusi_ttl_onsets, fill_value='extrapolate')
        factor = dt_ms/ttl_dt
        new_samples = np.arange(fusi_data.shape[0])*factor
        fusi_times = interpolator(new_samples)
        return fusi_times, fusi_data

    def fusi_get_outbrainroi_mask(self):
        '''
        '''
        probe_mask = self.fusi_get_probe_master_mask('probe00').astype(np.bool)
        zdim, xdim = probe_mask.nonzero()
        length = np.sqrt((zdim.max() - zdim.min())**2 +
                         (xdim.max() - xdim.min())**2)
        nvox = probe_mask.sum()
        depth = int(nvox/length)
        length = int(length)

        leftmost_side = int((probe_mask.shape[1] - length)/2.)
        outbrain_mask = self.fusi_get_outsidebrain_mask()
        outbrain_top = np.logical_not(outbrain_mask).nonzero()[0].min()
        outbrain_roi = np.zeros_like(outbrain_mask)
        outbrain_roi[outbrain_top - depth: outbrain_top,
                     leftmost_side:leftmost_side + length] = True
        return outbrain_roi

    def ephys_get_lfp(self, band, probe_name):
        '''
        '''

        flname = self.cortexlab_mk_filename(
            '%s_lfp_bands_v01.hdf' % probe_name, 'lfp')
        assert flname.exists()

        available_bands = ['alpha', 'beta', 'gamma', 'hgamma']
        assert band in available_bands
        lfp_band = readers.hdf_load(flname, band).T

        times = self.fusi_get_times()
        assert len(times) == lfp_band.shape[0]
        return times, lfp_band[:, :384]
