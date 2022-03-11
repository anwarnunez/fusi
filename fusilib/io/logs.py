'''
Code to read INI experiment logs.

Anwar O. Nunez-Elizalde (Jan, 2020)
'''

import os
import pathlib
import urllib
import configparser


def str2bool(string):
    '''Convert a `string` to a `bool` type
    '''
    val = string[:]
    val = True if string in ['True', 'true'] else val
    val = False if string in ['False', 'false'] else val
    val = None if string in ['None', 'none'] else val
    return val


def can_float(string):
    '''
    '''
    try:
        value = float(string)
        return True
    except:
        # we can't even float
        return False


def can_int(string):
    if not can_float(string):
        # no float implies no int
        return False
    try:
        test = int(string)
    except ValueError:
        return False

    # should be the same if int
    return float(string) == int(string)


def can_list(string):
    # must have comma separation
    ans = ',' in string
    return ans


def can_bool(string):
    '''Test whether `string` can map to `bool`
    '''
    booleables = ['True', 'true',
                  'False', 'false',
                  'None', 'none']
    return string in booleables


def str2type(string):
    clean_string = string.strip()

    if can_int(clean_string):
        value = int(clean_string)
    elif can_float(clean_string):
        value = float(clean_string)
    elif can_bool(clean_string):
        value = str2bool(clean_string)
    else:
        value = clean_string
    return value


def convert_string2data(string):
    '''
    '''
    if not can_list(string):
        value = str2type(string)
    else:
        substrings = string.split(',')
        value = []
        for substring in substrings:
            val = str2type(substring)
            value.append(val)
        # Clean ONE trailing comma
        if string[-1] == ',' and value[-1] == '':
            value = value[:-1]
    return value


def convert_uri2local(uripath):
    '''
    URI follows: `scheme://netloc/path`

    For samba shares:
        smb://domain.server.com/Subjects/CR017/2019-11-20/
        -> /DATA_ROOT/

    Parameters
    ----------
    uripath (str): <scheme>://<netloc>/<path>
        `netloc` follows the <server.domain.ext> convention
        e.g. domain.server.com

    Returns
    -------
    path (str): /domain/server/path

    Examples
    --------
    >>> convert_uri2local('smb://domain.server.com/Subjects/CR017/2019-11-20/ephys/')
    '/DATA_ROOT/CR017/2019-11-20/ephys/'
    >>> convert_uri2local('file:///DATA_ROOT/CR017/2019-11-20/ephys/')
    '/DATA_ROOT/CR017/2019-11-20/ephys/'
    >>> # Assumes path is given if `scheme` (e.g. `smb://`) is not provided
    >>> convert_uri2local('/DATA_ROOT/CR017/2019-11-20/ephys/')
    '/DATA_ROOT/CR017/2019-11-20/ephys/'
    '''
    url_object = urllib.parse.urlsplit(uripath)

    if url_object.scheme is None:
        # assume it's just a path and do nothing
        return url_object.path

    path = url_object.path
    if url_object.netloc:
        server, domain = url_object.netloc.split('.')[:2]
        path = path[1:] if path[0] == '/' else path
        path = os.path.join('/', domain, server, path)
    return path


def read_experiment_log(flname, show_section='globals', verbose=False, **kwargs):
    '''Read a log text file created using `INI` format

    Parameters
    ----------
    flname : str
        Path
    show_section : optional, str:
        Display contents for this section
    **kwargs: passed to configparser.ConfigParser

    Returns
    -------
    config : ConfigParser instance

    Examples
    --------
    >>> sample_config = """[globals]
    root = smb://domain.server.com/Subjects
    subject = CR017
    date = 2019-11-20
    path_session = ${globals:root}/${globals:subject}/${globals:date}
    """

    See
    ---
    https://en.wikipedia.org/wiki/INI_file
    https://docs.python.org/3/library/configparser.htm
    '''
    if not os.path.exists(flname): # check it exists
        raise IOError('Does not exist: %s'%flname)

    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation(), **kwargs)
    config.read(flname)
    if show_section in config.sections():
        for k in config[show_section]:
            if verbose:
                print('%s:%s ='%(show_section, k), config[show_section][k])
    if len(config.sections()) == 0:
        print('File is empty: %s'%flname)
    return config


def read_log_field_data(log_object, key):
    '''
    '''
    if '/' not in key:
        ## robustness checks
        # does request exist?
        assert key in log_object
        string = log_object[key]
        # make sure a value and not a section was requested
        if isinstance(string, configparser.SectionProxy):
            raise IOError('Requested data is a section, not a value: %s'%key)
        # Finally load data and convert to correct data type
        value = convert_string2data(string)
        return value

    first_key = key.split('/')[0]             # car
    other_keys = '/'.join(key.split('/')[1:]) # cdr
    return read_log_field_data(log_object[first_key], other_keys)
