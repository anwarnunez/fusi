'''
Deal with DeepLabCut eye tracking

Also, code to handle motion energy videography.

Anwar Nunez-Elizalde (June, 2021)
'''
import pathlib
from functools import wraps

import numpy as np
from scipy import signal
import pandas

from fusi.extras import readers
from fusi.config import DATA_ROOT


def convert_dlccsv2df(flname, verbose=True):
    '''Convert a CSV created by DeepLabCut as a pandas DataFrame

    Parameters
    ----------
    flname : str

    Returns
    -------
    df : pandas.DataFrame
    '''
    # The First row is info about the network
    # the 2nd and 3rd rows together define the unique column name
    colnames = np.loadtxt(flname, skiprows=1, max_rows=2, dtype='S', delimiter=',')
    colnames = np.asarray(['{}_{}'.format(a.decode(),b.decode()) \
                           for a,b in zip(colnames[0], colnames[1])])

    # The rest is the data
    data = np.loadtxt(flname, skiprows=3, delimiter=',')

    if verbose:
        print(colnames)
        print(colnames.shape)
        print(data.shape)

    # sanity
    assert data.shape[1] == colnames.shape[0]

    datdict = {name : dat for name, dat in zip(colnames, data.T)}
    if verbose:
        print([(k,v.shape) for k,v in datdict.items()])

    df = pandas.DataFrame.from_dict(datdict)
    return df


def cortexlab_filename2info(flname):
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


def fl2info(func):
    '''
    '''
    @wraps(func)
    def wrapper(first_arg, *args, **kwargs):
        if len(args) == 0:
            # this is a filename
            args = cortexlab_filename2info(first_arg)
        else:
            args = (first_arg,) + args
        return func(*args, **kwargs)

    #wrapper.__doc__ = func.__doc__
    return wrapper


@fl2info
def get_video_file(subject, date, block):
    '''Get the eye video file from the block

    Parameters
    ----------
    subject, date, block : str

    Returns
    -------
    videofl : pathlib.Path
    '''
    path = pathlib.Path(DATA_ROOT).joinpath(subject, date, block)
    video_pattern = '{date}_{block}_{subject}_eye.mj2'.format(date=date,
                                                              block=block,
                                                              subject=subject)

    videofl = path / video_pattern
    return videofl




@fl2info
def get_dlc_csv(subject, date, block,
                dlc_pattern='eyetrackingDLC_v01'):
    '''Get the CSV file from DLC analysis

    Parameters
    ----------
    subject, date, block : str

    Returns
    -------
    cvsfl : pathlib.Path
    '''
    path = pathlib.Path(DATA_ROOT).joinpath(subject, date, block)
    csv_pattern = '{date}_{block}_{subject}_{name}.csv'.format(date=date,
                                                               block=block,
                                                               subject=subject,
                                                               name=dlc_pattern)

    return path / csv_pattern


def get_dlc_df(*args, **kwargs):
    '''Get a DataFrame from DLC analysis

    Parameters
    ----------
    subject, date, block : str

    Returns
    -------
    df : pandas.DataFrame
    '''
    fl = get_dlc_csv(*args, **kwargs)
    return pandas.read_csv(fl)


@fl2info
def show_pupil_points(subject, date, block,
                      background=True, ax=None, **kwargs):
    '''
    '''
    df = get_dlc_df(subject, date, block, **kwargs)

    if ax is None:
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(nrows=3)

    ax = axes[0]
    if background is True:
        import moten
        vfl = str(get_video_file(subject, date, block))
        im = moten.io.video2grey(vfl, nimages=1)[0]
        ax.imshow(im, vmin=0, vmax=1, cmap='Greys_r')

    ax.plot(df['corner_left_x'], df['corner_left_y'], 'ko', alpha=0.1, label='corner left')
    ax.plot(df['corner_right_x'], df['corner_right_y'], 'ro', alpha=0.1, label='corner right')

    ax.plot(df['pupil_left_x'], df['pupil_left_y'], 'mo', alpha=0.1, label='pupil left')
    ax.plot(df['pupil_right_x'], df['pupil_right_y'], 'yo', alpha=0.1, label='pupil right')


    ax.plot(df['pupil_top_x'], df['pupil_top_y'], 'go', alpha=0.1, label='pupil top')
    ax.plot(df['pupil_bot_x'], df['pupil_bot_y'], 'co', alpha=0.1, label='pupil bottom')

    if background is not True:
        ax.invert_yaxis()

    ax.set_xlabel('left <-> right')
    ax.set_ylabel('down <-> up')
    ax.legend(loc='lower right')

    ax = axes[1]

    # pupil_size = df['pupil_bot_y'] - df['pupil_top_y']
    pupil_size = get_pupil_vsize(subject, date, block, **kwargs)
    FPS = 30
    time = np.arange(len(pupil_size))*(1./FPS)

    ax.plot(time, pupil_size)

    session_pupil_size = get_pupil_size(subject, date)
    ax.hlines(np.median(session_pupil_size), time.min(), time.max(), color='k')
    ax.hlines(np.percentile(session_pupil_size, 95), time.min(), time.max(), color='r', linestyle=':')

    ax.set_ylim(0, 30) if pupil_size.max() < 30 else ax.set_ylim(0, 50)
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Pupil size [pixels]')

    ax = axes[2]
    xcenter = (df['pupil_left_x'] + df['pupil_right_x'])/2.
    ycenter = (df['pupil_left_y'] + df['pupil_right_y'])/2.

    ax.plot(time, xcenter - np.median(xcenter), color='m')
    ax.plot(time, ycenter - np.median(ycenter), color='g')
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Pupil center [horiz, vert]')

    return axes


@fl2info
def get_pupil_vsize(subject, date, block, rectify=True, median_filter_window=29, **kwargs):
    '''
    '''
    df = get_dlc_df(subject, date, block, **kwargs)

    # bottom is a higher number
    pupil_size = df['pupil_bot_y'] - df['pupil_top_y']
    nframes = (pupil_size < 0).sum()
    if rectify and nframes > 0:
        # top of pupil is lower than bottom of pupil.
        # probably means the points are mistakenly flipped
        pupil_size = np.where(pupil_size > 0, pupil_size, pupil_size*-1)
        print('Rectified: %0.02fpct (nframes=%i)'%( 100*nframes/len(pupil_size), nframes))

    if median_filter_window is not None:
        pupil_size = signal.medfilt(pupil_size, median_filter_window)
    return pupil_size


def get_pupil_size(subject, date, dlc_pattern='eyetrackingDLC_v01', verbose=True):
    '''
    '''
    path = pathlib.Path(DATA_ROOT).joinpath(subject, date)
    csv_pattern = '*/{date}_*_{subject}_{name}.csv'.format(date=date,
                                                           subject=subject,
                                                           name=dlc_pattern)

    fls = path.glob(csv_pattern)
    pupil_size = []
    for fl in fls:
        if verbose: print(fl)
        block_pupil = get_pupil_vsize(fl)
        pupil_size.append(block_pupil)

    return np.hstack(pupil_size)




@fl2info
def get_moten_hdf(subject, date, block,
                  pattern='whiskers_session_totalmoten_v01'):
    '''Get the CSV file from DLC analysis

    Parameters
    ----------
    subject, date, block : str

    Returns
    -------
    cvsfl : pathlib.Path
    '''
    path = pathlib.Path(DATA_ROOT).joinpath(subject, date, block)
    hdf_pattern = '{date}_{block}_{subject}_{name}.hdf'.format(date=date,
                                                               block=block,
                                                               subject=subject,
                                                               name=pattern)

    return path / hdf_pattern


@fl2info
def get_whisker_moten_block(subject, date, block, pattern='whiskers_session_totalmoten_v01',
                            npcs=10, func=np.abs, key='temporal_pcs'):
    '''
    contents = {'spatial_pcs' : np.asarray(tot.decomposition_spatial_pcs).astype(np.float32),
               'temporal_pcs' : np.asarray(tot.decomposition_temporal_pcs).astype(np.float32),
               'eigenvals' : np.asarray(tot.decomposition_eigenvalues).astype(np.float32),
               'xtx' : roi_xtx.astype(np.float32),
               'example_original_image' : oim,
               'example_masked_image' : nim,
               'mask': mask,
               'nimages' : nframes}
    '''
    fl = get_moten_hdf(subject, date, block, pattern=pattern)
    tpcs = readers.hdf_load(str(fl), key)[:, :npcs]
    return func(tpcs)


def get_whisker_moten(subject, date, pattern='whiskers_session_totalmoten_v01',
                      npcs=10, func=np.abs, key='temporal_pcs', verbose=True):
    '''
    '''
    path = pathlib.Path(DATA_ROOT).joinpath(subject, date)
    pattern = '*/{date}_*_{subject}_{name}.hdf'.format(date=date,
                                                       subject=subject,
                                                       name=pattern)
    fls = path.glob(pattern)
    whiskers_moten = []
    for fl in fls:
        if verbose: print(fl)

        tpcs = readers.hdf_load(str(fl), key, verbose=False)[:, :npcs]
        whiskers_moten.append(func(tpcs))

    return np.vstack(whiskers_moten)


def get_whisker_covar(subject, date, pattern='whiskers_session_totalmoten_v01',
                      key='xtx'):
    '''
    '''
    path = pathlib.Path(DATA_ROOT).joinpath(subject, date)
    pattern = '*/{date}_*_{subject}_{name}.hdf'.format(date=date,
                                                       subject=subject,
                                                       name=pattern)
    fls = path.glob(pattern)
    whiskers_moten = 0
    nimages = 0
    for fl in fls:
        print(fl)

        tpcs = readers.hdf_load(str(fl), 'xtx', verbose=False)
        nims = readers.hdf_load(str(fl), 'nimages', verbose=False)
        whiskers_moten += tpcs/nims
        nimages += nims
    return nimages, whiskers_moten
