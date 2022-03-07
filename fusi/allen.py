import pandas
import numpy as np
from allensdk.core.reference_space_cache import ReferenceSpaceCache

from fusi.config import DATA_ROOT

reference_space_key = 'annotation/ccf_2017'
resolution = 25
rspc = ReferenceSpaceCache(
    resolution, reference_space_key, manifest='manifest.json')
# ID 1 is the adult mouse structure graph
tree = rspc.get_structure_tree(structure_graph_id=1)
structureid2name = tree.get_name_map()


TABLE = pandas.read_csv(
    f'{DATA_ROOT}/extras/structure_tree_safe_2017.csv')

HPC = 'Hippocampal formation'
V1 = 'Visual areas'


def find_area(area_name, table=TABLE):
    matches = {table.iloc[idx].id: name for idx, name in enumerate(
        table.name) if area_name in name.lower()}
    return matches


def get_allenccf_parent_names(area_table_index, table=TABLE):
    parent_structure_id = table.iloc[area_table_index].parent_structure_id
    if np.isnan(parent_structure_id):
        parent_name = 'root'
    else:
        parent_name = structureid2name[parent_structure_id]
    return parent_name


def mk_area_mask_from_aligned(allenccf_areas, candidate, verbose=True):
    '''
    V1 = 'Visual areas'
    HPC = 'Hippocampal formation'
    VISp = 'Primary visual area'
    '''
    if candidate == 'V1':
        candidate = V1  # global name
    elif candidate == 'HPC':
        candidate = HPC  # global names
    elif candidate == 'VISp':
        candidate = 'Primary visual area'

    candidateid = tree.get_structures_by_name([candidate])[0]['id']

    mask = np.zeros(allenccf_areas.shape, dtype=np.bool)
    uareas = np.unique(allenccf_areas)
    uareas = uareas[uareas != 0]  # remove root node
    for area_index in uareas:
        areaid = TABLE.iloc[area_index].id
        if tree.structure_descends_from(areaid, candidateid):
            if verbose:
                print('%s is in %s' %
                      (structureid2name[areaid], structureid2name[candidateid]))
            mask[allenccf_areas == area_index] = True
    return mask
