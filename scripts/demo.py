import numpy as np
from scipy.stats import zscore
from matplotlib import pyplot as plt

import fusi.config
# Enter the path to the downloaded "Subjects" directory
data_location = '/path/to/extracted/data/Subjects'
data_location = '/store/fast/fusi_dataset_export/Subjects/'
fusi.config.set_dataset_path(data_location)

from fusi import handler2 as handler
from fusi import metahelper


#############################
# Setup
#############################
# Session data for this subject
subject = handler.MetaSession('CR017', '2019-11-13', verbose=False)

# Iterate through all blocks for this session
for idx, subject_block in enumerate(subject.generate_analysis_blocks()):
    print(subject_block)

#############################
# Loading data from a block
#############################
# Loda a data block explicitly
subject_block = handler.MetaBlock('CR017', '2019-11-13', block_name='3')

# Load fUSI data
original_fusi_times, fusi_data = subject_block.fusi_get_data()

# Load ephys data
probe_name = 'probe00'
probe = subject_block.ephys_get_probe_spikes_object(probe_name)

# Extract fUSI and ephys masks
#############################
# NOTE:
# * slice_fusi_mask: Mask of voxels at site of ephys probe insertion
# * slice_probe_mask: Location of probe depths spanning the fUSI slice
_, slice_fusi_mask, slice_probe_mask = subject_block.fusi_get_slice_probedepth_and_voxels(
    probe_name)
# Sanity: Exclude all voxels outside the brain
brain_mask = np.logical_not(subject_block.fusi_get_outsidebrain_mask())
block_mask = np.logical_and(brain_mask, slice_fusi_mask)

# Get a fUSI trace for the voxels within the mask
fusi_dt = 0.300  # fUSI sampling rate in [sec]
fusi_times, fusi_trace = metahelper.detrend_fusi(original_fusi_times,
                                                 fusi_data,
                                                 mask=block_mask,
                                                 temporal_filter='none',
                                                 fusi_dt=fusi_dt,
                                                 trim_after_beg=10,
                                                 trim_after_end=10)


# Build MUA matrix for units intersecting the fUSI slice
mua_probe_widthum = 200
probe_min_depthum, probe_max_depthum = (slice_probe_mask*1000).astype(int)
n_mua_units, mua_matrix = probe.time_locked_mua_matrix(
    fusi_times,
    dt=fusi_dt,
    good_clusters=probe.good_clusters,
    cluster_depths=probe.cluster_depths,
    min_depthum=probe_min_depthum,
    max_depthum=probe_max_depthum)

# Get the firing rate trace
firing_rate = metahelper.normalize_mua_matrix(mua_matrix)

# Plotting
fig, ax = plt.subplots()
ax.plot(fusi_times, zscore(fusi_trace), label='fUSI')
ax.plot(fusi_times, zscore(firing_rate), label='F.R.')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Activity [z-score]')
ax.legend(loc='best')
ax.set_title(subject_block)

plt.show()
