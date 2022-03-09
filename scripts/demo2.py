import numpy as np
from scipy.stats import zscore
from matplotlib import pyplot as plt

import fusi.config
# Enter the path to the downloaded "Subjects" directory
data_location = '/path/to/extracted/data/Subjects'
fusi.config.set_dataset_path(data_location)

from fusi import handler2 as handler
from fusi import metahelper


#############################
# V1
#############################
subject_block = handler.MetaBlock('CR017', '2019-11-13', block_name='3')

area_name = 'V1'
probe_name_rh = 'probe00'
probe_name_lh = 'probe01'
hrf_times, hrf = metahelper.get_population_hrf(area_name,
                                               subject_block.task_name,
                                               negation=True,  # negation means get the HRF from the OTHER stimulus conditions
                                               normalize=True,
                                               )

fusi_times, fusi_lh, fr_lh = metahelper.get_fusi_roi_mask_within_npxprobe(
    subject_block,
    probe_name_lh,
    area_name,
    hrf_convolve=hrf)

fusi_times, fusi_rh, fr_rh = metahelper.get_fusi_roi_mask_within_npxprobe(
    subject_block,
    probe_name_rh,
    area_name,
    hrf_convolve=hrf)


# Plotting
fig, axes = plt.subplots(ncols=2)
V1_data = dict(LH=(fusi_times, fusi_lh, fr_lh),
               RH=(fusi_times, fusi_lh, fr_lh))

for idx, (hemi, (times, fusi_trace, fr)) in enumerate(V1_data.items()):
    ax = axes[idx]
    ax.plot(times, zscore(fusi_trace), label='fUSI')
    ax.plot(times, zscore(fr), label='F.R.')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Activity [z-score]')
    ax.legend(loc='best')
    ax.set_title(hemi)
fig.suptitle(area_name)


#############################
# Hippocampus
#############################
subject_block = handler.MetaBlock('CR017', '2019-11-13', block_name='9')

area_name = 'HPC'
probe_name_rh = 'probe00'
probe_name_lh = 'probe01'
hrf_times, hrf = metahelper.get_population_hrf(area_name,
                                               subject_block.task_name,
                                               negation=True,  # negation means get the HRF from the OTHER stimulus conditions
                                               normalize=True,
                                               )

fusi_times, fusi_lh, fr_lh = metahelper.get_fusi_roi_mask_within_npxprobe(
    subject_block,
    probe_name_lh,
    area_name,
    hrf_convolve=hrf)

fusi_times, fusi_rh, fr_rh = metahelper.get_fusi_roi_mask_within_npxprobe(
    subject_block,
    probe_name_rh,
    area_name,
    hrf_convolve=hrf)


# Plotting
fig, axes = plt.subplots(ncols=2)
HPC_data = dict(LH=(fusi_times, fusi_lh, fr_lh),
                RH=(fusi_times, fusi_lh, fr_lh))

for idx, (hemi, (times, fusi_trace, fr)) in enumerate(HPC_data.items()):
    ax = axes[idx]
    ax.plot(times, zscore(fusi_trace), label='fUSI')
    ax.plot(times, zscore(fr), label='F.R.')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Activity [z-score]')
    ax.legend(loc='best')
    ax.set_title(hemi)
fig.suptitle(area_name)
