'''
io stuff
'''

import os
from os import path as op
from os.path import splitext, exists as pexists
import re

import sys
if sys.version_info.major == 2:
    import pickle
else:
    import pickle as cPickle


try:
    from io import StringIO
except ImportError:
    from io import BytesIO as StringIO

from dateutil.tz import tzlocal

import hashlib
import shutil
from warnings import warn

import h5py
import tables

import numpy as np
from scipy.io import loadmat as sioloadmat


def _hdf_field(hdfobj, field, mask=False):

    data = hdfobj[field]

    if (mask is False) or (mask is None):
        try:
            output = data[:]
        except TypeError:
            output = data
        except ValueError:
            output = np.asarray(data)

    elif len(data.shape) > 1:
        output = data[..., mask]
    else:
        output = data[mask]
    return output


def hdf_load(fl, field=None, mask=False,
             verbose=False, cache=False):
    '''
    Load HDF file.

    Parameters
    ----------
    file (str)
         Path to the HDF file to load
    field (str, list)
         Field(s)/group(s) to load
    mask (1D array)
         This is used to mask the data being loaded.
         If the data is 2D or more, the mask
         is applied on the last dimension (ie data[...,mask]).
         Otherwise, the mask is applied to the data
         (ie data[mask]).

    '''
    if not op.exists(fl):
        raise IOError('"%s" does not exist!' % fl)

    if verbose:
        print(('Loading %s...' % fl))

    if cache:
        fl = local_cache_file(fl)

    with h5py.File(fl, 'r') as f:
        if field is None:
            t = []
            for fk in list(f.keys()):
                # for v7.3 matlab files
                try:
                    if isinstance(f[fk], h5py.Group):
                        fkk = ','.join(['{}/%s'.format(fk) %
                                       kn for kn in f[fk].keys()])
                        print(fk, (fkk))
                    else:
                        fkk = f[fk].shape
                        print(fk, (fkk), f[fk].dtype)
                except:
                    print(fk)
                t.append(fk)  # (fk, f[fk].shape, f[fk].dtype))
            return t

        elif isinstance(field, str) or isinstance(field, str):
            output = _hdf_field(f, field, mask=mask)

        elif isinstance(field, list):
            output = []
            for k in field:
                d = _hdf_field(f, k, mask=mask)
                output.append(d)
    return output


def tab_field(hdf_file, field=None, verbose=True):
    '''
    Wrapper to `__getattr__` for a table object
    '''
    if verbose:
        print(('Loading %s...' % hdf_file))
    hdf_file = tables.open_file(hdf_file, 'r')

    if field is None:
        fields = hdf_file.root._f_listNodes()
        if verbose:
            print(fields)
        field_names = [t.name for t in fields]
        hdf_file.close()
        return field_names

    if type(field) == list:
        dat = []
        for fld in field:
            dats = hdf_file.root.__getattr__(fld)[:]  # .copy()
            dat.append(dats)
    else:
        try:
            dat = hdf_file.root.__getattr__(field)[:]  # .copy()
        except IndexError:
            dat = hdf_file.root.__getattr__(field)

    hdf_file.close()
    return dat


def update_hdf(flname, datadict, overwrite=False,
               compression='gzip', **kwargs):
    '''
    '''
    with h5py.File(flname, 'a') as hdf:
        contents = list(hdf.keys())
        for key, value in datadict.items():
            if key in contents:
                if overwrite is True:
                    print('Overwriting "%s"' % key)
                    hdf.__delitem__(key)
                else:
                    print('Already exists: "%s" in %s' % (key, flname))
                    continue

            if isinstance(value, dict):
                grp = hdf.create_group(key)
                for kk, vv in value.items():
                    vv = np.atleast_1d(vv)
                    grp.create_dataset(kk,
                                       data=np.atleast_1d(vv),
                                       compression=compression,
                                       **kwargs)
                    print('Saving: %s/%s' % (key, kk), vv.shape)
            else:
                value = np.atleast_1d(value)
                hdf.create_dataset(key,
                                   data=np.atleast_1d(value),
                                   compression=compression,
                                   **kwargs)
                print('Saving: %s' % key, value.shape)
    print('Updated: %s' % flname)


def save_hdf(flname, datadict, compression='gzip', **kwargs):
    '''
    '''
    with h5py.File(flname, mode='w') as hdf:
        for key, value in datadict.items():
            if isinstance(value, dict):
                grp = hdf.create_group(key)
                for kk, vv in value.items():
                    vv = np.atleast_1d(vv)
                    grp.create_dataset(kk,
                                       data=np.atleast_1d(vv),
                                       compression=compression,
                                       **kwargs)
                    print('Saving: %s/%s' % (key, kk), vv.shape)
            else:
                value = np.atleast_1d(value)
                hdf.create_dataset(key,
                                   data=np.atleast_1d(value),
                                   compression=compression,
                                   **kwargs)
                print('Saving: %s' % key, value.shape)
    print('Wrote: %s' % flname)


def load_hdf(hdf_file, key):
    '''Load data from an HDF5 file
    '''
    assert os.path.exists(hdf_file)

    with h5py.File(hdf_file, 'r') as hfl:
        hdf_content = hfl.keys()
        assert key in hdf_content
        return np.asarray(hfl[key])


def hdf_dump(filename, filedict, mode='w'):
    """Saves the variables in [filedict] in a hdf5 table file at [filename].
    """
    with tables.open_file(filename, mode=mode, title="save_file") as hf:
        for vname, var in list(filedict.items()):
            if not isinstance(var, np.ndarray):
                hf.create_array("/", vname, np.asarray(var))
            else:
                hf.create_array("/", vname, var)
        hf.close()
    return


def tab2dic(table_obj):
    '''
    Get all the nodes from a python tables object
    If `table_obj` is a string, we will load the file first
    '''
    if type(table_obj) == str:
        print(('Loading %s...' % table_obj))
        table_obj = tables.open_file(table_obj, 'r')

    obs = table_obj.root._f_listNodes()
    tabdict = {}
    for ob in obs:
        try:
            field_name = ob.name
        except tables.NoSuchNodeError:
            continue

        print(('...Getting `%s` ...' % field_name))
        tabdict[field_name] = tab_field(table_obj, field_name)
    table_obj.close()
    return tabdict


def hdf2dic(f=None, group=None, parent=None):
    ''' This works well for getting all the data
    from HDF5 matlab files that are weird'''
    if parent is not None:
        group = parent + '/' + group
        parent = None
    if type(f) == str:
        f = hdf.File(f, 'r')

    # Try to get some keys
    try:
        if group is None:
            subgroups = [t for t in list(f.keys()) if '#' not in t]
        else:
            subgroups = [t for t in list(f[group].keys()) if '#' not in t]
        # We are not at a bottom level, so keep going
        outdict = {}
        for sg in subgroups:
            out = hdf2dic(f=f, group=sg, parent=group)
            outdict[sg] = out
        return outdict
    # We are the bottom level
    except:
        try:
            sgdata = np.array(f[f[group][0].item()])
            return sgdata

        except ValueError:
            sgdata = np.array(f[group])
            return sgdata

        except:
            print((group, parent))
            pass


if __name__ == '__main__':
    pass
