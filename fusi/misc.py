'''
Random stuff.

Critically, these functions are not for data analysis.
Those should be in `fusi.utils`

Functions should be as self-contained as possible.
Such that they can be moved easily into other modules.

Anwar O. Nunez-Elizalde (Jan, 2020)
'''
import os
import pathlib
import urllib
from datetime import datetime



class DotDict(dict):
    '''A subclass of dictionary with an additional dot syntax.

    Code copied from pykilosort2 (`Bunch`)
    '''
    def __init__(self, *args, **kwargs):
        super(type(self), self).__init__(*args, **kwargs)
        self.__dict__ = self

    def copy(self):
        return DotDict(super(type(self), self).copy())


def string_format_keys(mystring):
    '''
    '{path}/my_analysis{version}' -> ['path', 'version']
    '{same}/my_analysis{same}' -> ['same', 'same']
    '{some}/my_analysis{}' -> ['some', '']
    '''
    import string
    formatted = string.Formatter().parse(mystring)
    keys = [t[1] for t in formatted if t[1] is not None]
    return keys


##############################
# date-stringify-ing
###############################

def date_cxlab2dateob(date_string):
    '''
    '''
    return datetime.strptime(date_string, '%Y-%m-%d')


def date_tuple2yyyymmdd(date_tuple):
    '''(2020, 1, 31) -> ('2020', '01', '31')
    '''
    dateob = datetime.strptime(str(date_tuple),
                                        '(%Y, %m, %d)')
    ymd = (dateob.year, dateob.month, dateob.day)
    date = '%04i,%02i,%02i'%ymd
    return date.split(',')

def date_cortexlab2sensible(date_string):
    '''
    '''
    tup = date_cortexlab2tuple(date_string)
    sens = date_tuple2sensible(tup)
    return sens


def date_tuple2cortexlab(date_tuple):
    '''(2020, 1, 31) -> '2020-01-31'
    '''
    dateob = datetime.strptime(str(date_tuple),
                                        '(%Y, %m, %d)')

    ymd = (dateob.year, dateob.month, dateob.day)
    date = '%04i-%02i-%02i'%ymd
    return date

def date_tuple2sensible(date_tuple):
    '''(2020, 1, 31) -> '20200131'
    '''
    dateob = datetime.strptime(str(date_tuple),
                                        '(%Y, %m, %d)')

    ymd = (dateob.year, dateob.month, dateob.day)
    date = '%04i%02i%02i'%ymd
    return date

def date_tuple2number(date_tuple):
    '''(2020, 1, 31) -> 20200131 # INT data-type
    '''
    dateob = datetime.strptime(str(date_tuple),
                                        '(%Y, %m, %d)')
    ymd = (dateob.year, dateob.month, dateob.day)
    date = '%04i%02i%02i'%ymd
    date = int(date)
    return date

def date_cortexlab2tuple(datestring):
    '''
    '2019-01-31' -> (2019, 1, 13)
    '2019-1-31' -> (2019, 1, 13)
    '''
    assert isinstance(datestring, str)
    dateob = datetime.strptime(datestring,
                               '%Y-%m-%d')
    time_tuple = dateob.timetuple()
    return time_tuple[:3]

def date_number2tuple(datestring):
    '''20191231 -> (2019,12,13)
    '''
    assert isinstance(datenumber, int)
    dateob = datetime.strptime(str(datenumber),
                               '%Y%m%d')
    time_tuple = dateob.timetuple()
    return time_tuple[:3]


def date_sensible2tuple(ymd):
    '''
    '20191231' -> (2019,12,13)
    20191231 -> (2019,12,13)
    '''
    dateob = datetime.strptime(str(ymd),
                               '%Y%m%d')
    time_tuple = dateob.timetuple()
    return time_tuple[:3]

def date_cortexlab2dateob(datestring):
    '''
    '''
    assert isinstance(datestring, str)
    dateob = datetime.strptime(datestring,
                               '%Y-%m-%d')
    return dateob


##############################
# reading data strings into python
##############################

def str2bool(string):
    '''Convert a `string` to a `bool` type
    '''
    val = string[:]
    val = True if string in ['True', 'true'] else val
    val = False if string in ['False', 'false'] else val
    val = None if string in ['None', 'none'] else val
    return val

def can_float(string):
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
    '''Convert string to correct data type.

    Notes
    -----
    '1.0'   -> 1.0              # float
    '1'     -> 1                # int
    'True'  -> True             # bool
    'false' -> False            # bool
    'none'  -> None             # bool
    ''      -> ''               # str
    'asdf'  -> 'asdf'           # str
    '''
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
    '''Convert string to correct data type.

    This function can handle strings and string-lists.

    Examples
    --------
    >>> convert_string2data('1')  # parsed as int
    1
    >>> convert_string2data('1,') # parsed as list
    [1]
    >>> convert_string2data('1,2,3,4,')
    [1, 2, 3, 4]
    >>> # multiple data types within list
    >>> string = "1.0, 1, 2., False, asdf, none"
    >>> convert_string2data(string)
    [1.0, 1, 2.0, False, 'asdf', None]
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

##############################
# URI paths
##############################

def uri_convert_uri2local(uripath):
    '''
    URI follows: `scheme://netloc/path`

    For samba shares:
        smb://server.domain.com/Subjects/CR017/2019-11-20/
        -> /DATA_ROOT/CR017/2019-11-20/

    Parameters
    ----------
    uripath (str): <scheme>://<netloc>/<path>
        `netloc` follows the <server.domain.ext> convention
        e.g. server.domain.com

    Returns
    -------
    path (str): /domain/server/path

    Examples
    --------
    >>> uri_convert_uri2local('smb://server.domain.com/Subjects/CR017/2019-11-20/ephys/')
    '/DATA_ROOT/CR017/2019-11-20/ephys/'
    >>> uri_convert_uri2local('file:///DATA_ROOT/CR017/2019-11-20/ephys/')
    '/DATA_ROOT/CR017/2019-11-20/ephys/'
    >>> # Assumes path is given if `scheme` (e.g. `smb://`) is not provided
    >>> uri_convert_uri2local('/DATA_ROOT/CR017/2019-11-20/ephys/')
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


def uri_convert_local2uri(local_path, scheme='smb'):
    '''
    URI follows: `scheme://netloc/path`
    Local paths: `/netloc/path`

    For samba shares:
        /DATA_ROOT/CR017/2019-11-20/
        -> smb://server.domain.com/Subjects/CR017/2019-11-20/

    '''
    ext = '.net'
    root, server, share = local_path.split('/')[1:4]
    path = '/'.join(local_path.split('/')[4:])
    pattern = '{scheme}://{server}.{root}{ext}/{share}/{path}'
    return pattern.format(scheme=scheme,
                          server=server,
                          root=root,
                          share=share,
                          ext=ext,
                          path=path)


def cortexlab_subject_localpath(subject,
                                server='server.domain.com',
                                share='Subjects'):
    uri = 'smb://{}/{}/{}'.format(server,share,subject)
    local_path = uri_convert_uri2local(uri)
    return local_path
