from collections import defaultdict as ddic
import numpy as np

from fusilib import align as _align
from fusilib import handler2 as handler


probe2hemisphere = {'CR017' : {'2019-11-13' : {'probe00' : 'RH',
                                               'probe01' : 'LH'},
                               '2019-11-14' : {'probe00' : 'RH',
                                               'probe01' : 'LH'}},
                    'CR020' : {'2019-11-20' : {'probe01' : 'LH'},
                               '2019-11-21' : {'probe01' : 'LH'},
                               '2019-11-22' : {'probe01' : 'LH'}},
                    'CR019' : {'2019-11-26' : {'probe00' : 'LH'},
                               '2019-11-27' : {'probe00' : 'LH'}},
                    'CR022' : {'2020-10-07' : {'probe00' : 'LH',
                                               'probe01' : 'RH'},
                               '2020-10-11' : {'probe00' : 'LH',
                                               'probe01' : 'RH'}},
                    'CR024' : {'2020-10-29' : {'probe00' : 'LH',
                                               'probe01' : 'RH'},
                    }
}


subject_sessions = {'CR017' : ['2019-11-13', '2019-11-14'],
                    'CR020' : ['2019-11-20', '2019-11-21', '2019-11-22'],
                    'CR019' : ['2019-11-26', '2019-11-27'],
                    'CR022' : ['2020-10-07', '2020-10-11'],
                    'CR024' : ['2020-10-29']}


ephys_roi_markers = {'CR017': {'2019-11-13': {'probe00': {'V1a': [2400, 2900],
                                                            'V1b': [2000, 2400],
                                                            'V1': [2000, 2900],
                                                            'X1': [1200, 1900],
                                                            'X2': [400, 1000],
                                                            'XX': [400, 1900],
                                                            },
                                                'probe01': {'V1a': [2000, 2400],
                                                            'V1b': [1700, 2000],
                                                            'V1': [1700, 2400],
                                                            'X1': [800, 1600],
                                                            'X2': [100,  600],
                                                            'XX': [100, 1600],
                                                            },
                                                },
                                '2019-11-14': {'probe00': {'V1a': [2500, 3100],
                                                            'V1b': [2200, 2500],
                                                            'V1': [2200, 3100],
                                                            'X1': [1400, 2000],
                                                            'X2': [600, 1200],
                                                            'X3': [0,  300],
                                                            'XX': [0, 2000],
                                                            },
                                                'probe01': {'V1a': [2600, 3000],
                                                            'V1b': [2100, 2600],
                                                            'V1': [2100, 3000],
                                                            'X1': [1300, 1900],
                                                            'X2': [100,  900],
                                                            'XX': [100, 1900],
                                                            },
                                                },
                                },

                        'CR019': {'2019-11-26': {'probe00': {'V1a': [2500, 2800],
                                                            'V1b': [2200, 2500],
                                                            'V1': [2200, 2800],
                                                            'X1': [700, 2100],
                                                            'X2': [300,  600],
                                                            },
                                                },
                                '2019-11-27': {'probe00': {'V1': [2000, 2500],
                                                            'X1': [800, 1800],
                                                            'X2': [0,  700],
                                                            },
                                                },
                                },

                        'CR020': {'2019-11-20': {'probe01': {'V1a': [2700, 3100],
                                                            'V1b': [2300, 2700],
                                                            'V1': [2300, 3100],
                                                            'X1': [1800, 2300],
                                                            'X2': [500, 1100],
                                                            },
                                                },
                                '2019-11-21': {'probe01': {'V1a': [2600, 3000],
                                                            'V1b': [2100, 2600],
                                                            'V1': [2100, 3000],
                                                            'X1': [1200, 1700],
                                                            'X2': [100,  400],
                                                            },
                                                },
                                '2019-11-22': {'probe01': {'V1a': [2500, 3000],
                                                            'V1b': [2200, 2500],
                                                            'V1': [2200, 3000],
                                                            'X1': [1800, 2200],
                                                            'X2': [900, 1600],
                                                            'X3': [200,  600],
                                                            },
                                                },
                                },
                        'CR022': {'2020-10-07': {'probe00': {'V1': [2100, 3300],  # LH
                                                            'X1': [700, 1800],
                                                            'XX': [700, 1800],
                                                            },
                                                'probe01': {'V1': [1900, 3300],  # RH
                                                            'X1': [300, 1700],
                                                            'XX': [300, 1700],
                                                            },
                                                },
                                '2020-10-11': {'probe01': {'V1': [2800, 3700],  # RH
                                                            'X1': [1100, 2600],
                                                            'XX': [1100, 2600],
                                                            },
                                                'probe00': {'V1': [2200, 3500],  # LH
                                                            'X1': [800, 1900],
                                                            'XX': [800, 1900],
                                                            },
                                                },
                                },
                        'CR024': {'2020-10-29': {'probe00': {'V1': [3100, 3700],  # LH
                                                            'X1': [1600, 2300],
                                                            'XX': [1600, 2300],
                                                            },
                                                'probe01': {'V1': [2700, 3800],  # RH
                                                            'X1': [700, 2600],
                                                            },
                                                },
                                }
}




automatic_probe_xyz_positions = {}
_probe_depths = {}
for sub, vv in ephys_roi_markers.items():
    automatic_probe_xyz_positions[sub] = ddic(dict)
    _probe_depths[sub] = ddic(dict)
    for sess, vvv in vv.items():
        for probe, rois in vvv.items():
            # find highest cluster in probe
            probe_depth_mm = np.asarray([max(r) for r in rois.values()]).max()/1000.0
            # find penetrantion inclination angle
            subject = handler.MetaSession(sub, sess)
            angle_inclination  = subject.log_load_section('%s_insertion'%probe, 'p')
            is_RH = -1 if subject.log_load_section('%s_insertion'%probe, 'hemisphere') == 'RH' else 1
            if 0: print(subject, angle_inclination, probe_depth_mm)
            automatic_probe_xyz_positions[sub][sess][probe] = _align.estimate_probe_xyz_from_angles(
                angle_inclination+90, angle_azimuth=90 - 45*is_RH,
                probe_depth_mm=probe_depth_mm)
            _probe_depths[sub][sess][probe] = (angle_inclination, probe_depth_mm)

# arguments are :
# inclination angle, width of projection (from fUSi), and probe depths
probe_xyz_positions = {
    'CR017': {
        '2019-11-13': {'probe00': _align.estimate_probe_xyz_for_probe(40, 1.3, 2.9, dv_projection_mm=2.0),
                       'probe01': _align.estimate_probe_xyz_for_probe(44, -1.1, 2.9, dv_projection_mm=2.2),
                       },
        '2019-11-14': {'probe00': _align.estimate_probe_xyz_for_probe(40, 1.7, 3.1, dv_projection_mm=2.2),
                       'probe01': _align.estimate_probe_xyz_for_probe(44, -1.0, 3.0, dv_projection_mm=2.3),
                       },
    },
    'CR019': {
        '2019-11-26': {'probe00': _align.estimate_probe_xyz_for_probe(50, -0.85, 2.8, dv_projection_mm=1.9) + np.asarray([0, 0.3, 0.0])},
        '2019-11-27': {'probe00': _align.estimate_probe_xyz_for_probe(50, -0.75, 2.5, dv_projection_mm=1.8) + np.asarray([0, 0.3, -0.3])},
    },
    'CR020': {
        '2019-11-20': {'probe01': _align.estimate_probe_xyz_for_probe(40, -1.1, 3.1, dv_projection_mm=1.75)},
        '2019-11-21': {'probe01': _align.estimate_probe_xyz_for_probe(44, -0.9, 3.0, dv_projection_mm=2)},
        '2019-11-22': {'probe01': _align.estimate_probe_xyz_for_probe(44, -1.1, 3.0, dv_projection_mm=1.8)},
    },
    'CR022': {
        '2020-10-07': {'probe01': _align.estimate_probe_xyz_for_probe(30,  1.5, 3.3, dv_projection_mm=1.7),
                       'probe00': _align.estimate_probe_xyz_for_probe(40, -1.2, 3.3, dv_projection_mm=1.7)},
        '2020-10-11': {'probe01': _align.estimate_probe_xyz_for_probe(30,  1.0, 3.8, dv_projection_mm=1.7),
                       'probe00': _align.estimate_probe_xyz_for_probe(30, -0.7, 3.5, dv_projection_mm=1.7)},
    },
    'CR024': {
        '2020-10-29': {
            'probe00': _align.estimate_probe_xyz_for_probe(30, -1.2, 3.8, dv_projection_mm=1.7),
            'probe01': _align.estimate_probe_xyz_for_probe(30,  1.2, 3.7, dv_projection_mm=1.7),
        }
    },
}

__all__ = [probe_xyz_positions,
           automatic_probe_xyz_positions,
           _probe_depths]
