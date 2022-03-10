import time
import warnings

import numpy as np
from scipy import ndimage

import nibabel as nib

from fusi.utils import hex2rgba

def cartesian2spherical(xyz):
    '''
    Returns
    -------
    radius : scalar
    inclination_deg : scalar
    azimuth_deg : scalar
    '''
    x,y,z = xyz
    radius = np.sqrt(np.sum(np.asarray([x,y,z])**2))
    angle_inclination = np.arccos(z/radius)
    angle_azimuth = np.arctan2(y,x)
    return radius, np.rad2deg(angle_inclination), np.rad2deg(angle_azimuth)


def estimate_probe_depth_from_coord(coord_mm,
                                    xyz_probe_tip,
                                    coord_offset_mm=0.0,
                                    xyz_axis=1,
                                    check_inside=True,
                                    verbose=False,
                                    ):
    '''Find probe depth that a position along cartisian axis (x,y,or z).

    Parameters
    ----------
    coord_mm : scalar, [mm]
        Position along axis of interest
    xyz_axis : int,
        Defaults to 1:yaxis:AP.
        0 : x-axis (ML), 1: y-axis (AP), 2: z-axis (DV)
    xyz_probe_tip : np.ndarra [mm], (3,)
        <x,y,z> [mm] vector of probe tip location.
    coord_offset : scalar, [mm]
        Probe offset. The offset will be subracted from the coordinate
        For AP, it is the distance from probe insertion to y=0
        e.g. AP y-coord=0 and offset 0.2 makes it such that y-coord=-0.2.

    Returns
    --------
    probe_depth : scalar, [mm]
        Position in probe at coordinate of interest.
        Convention where 0[mm] is top of probe.
        If the probe depth is outside the brain or beyond the tip,
        then the values are returned as negative.
    position_in_xyz : np.ndarray [mm], (3,)
        Position of probe at coordinate of interest
    '''
    scale = (coord_mm - coord_offset_mm)/xyz_probe_tip[xyz_axis]
    position_in_xyz = xyz_probe_tip*scale
    probe_depth = np.sqrt(np.sum(xyz_probe_tip**2))
    position_depth = np.sqrt(np.sum(position_in_xyz**2))
    depth_in_probe = probe_depth - position_depth*np.sign(position_in_xyz[xyz_axis])
    if verbose: print(position_in_xyz, position_depth, probe_depth)
    if depth_in_probe > probe_depth or depth_in_probe < 0:
        warnings.warn('Position is too long! %0.04f[mm]>%0.04f[mm]'%(probe_depth - depth_in_probe, probe_depth))
        if check_inside:
            raise ValueError('Position is too long! %0.04f[mm]>%0.04f[mm]'%(probe_depth - depth_in_probe, probe_depth))
        depth_in_probe *= -1
        position_depth *= -1
    return position_depth, position_in_xyz


def estimate_probe_xyz_from_angles(angle_inclination,
                                   angle_azimuth=45,
                                   probe_depth_mm=3.84):
    '''Estimate location of probe in cartesian coordinates

    Convention is in spherical coordinates and insertion site is origin (0,0,0).

    Notes
    -----
    For a manipulator with 30[deg] downward inclination,
    a probe inserted RH pointing towards the midline at 45[deg]:
        * angle_inclination = 90+30 # [deg] b/c 0[degs] points up
        * angle_azimuth = 90+45 #[deg] b/c 45[deg] points towards the right of the brain

    For a manipulator with 30[deg] downward inclination
    a probe inserted LH pointing towards the midline at 45[deg]:
        * angle_inclination = 90+30 # [deg] b/c 0[degs] points up
        * angle_azimuth = 45 #[deg] 45[deg] points towards the right of the brain (towards midline)

    Parameters
    ----------
    angle_inclination : scalar, [deg]
        Inclination in spherical coordinates (0[deg] points up).
        NB: For downward inclinations, add 90[deg] to manipulator setting.
    angle_azimuth : scalar, [deg]
        Azimuth in spherical coordinates (0[deg] points right)
        NB: For typical azimuths pointing towards midline:
            RH: 90 + azimuth [deg] if in RH
            LH: 90 - azimuth [deg] if in LH
    probe_depth_mm : scalar, [mm]
        Size of probe inside of brain

    Returns
    -------
    xyz_coords : np.ndarray, (3,)
        Position of probe tip in cartesian coordinates.
        Convention:
            x: right(+)/left(-)
            y: anterior(+)
            z: dorsal(+)/ventral(-)
        Because insertions typically pointing down, z is typically negative and -x is LH.
    '''
    xpos = probe_depth_mm*np.sin(np.deg2rad(angle_inclination))*np.cos(np.deg2rad(angle_azimuth))
    ypos = probe_depth_mm*np.sin(np.deg2rad(angle_inclination))*np.sin(np.deg2rad(angle_azimuth))
    zpos = probe_depth_mm*np.cos(np.deg2rad(angle_inclination))
    return np.asarray([xpos,ypos,zpos])

def estimate_probe_xyz_for_probe(angle_downward_inclination,
                                 xwidth_mm,
                                 probe_depth_mm,
                                 dv_projection_mm=None,
                                 angle_azimuth_nominal=45,
                                 verbose=False,
                                 **kwargs):
    '''
    xwidth_mm is signed -LH, +RH

    Parameters
    ----------
    angle_downward_inclination : float-like, [deg]
        Angle from the axial plane downwards
    xwidth_mm : float-like [mm]
        Width of fUSI probe 2D projection
        If negative, the probe is assumed to be in left hemisphere.
    probe_depth_mm : float-like [mm]
        Depth of probe

    Returns
    -------
    xyz : np.ndarray (3,)
       Position of probe tip in mm.
    '''
    right_hemisphere = np.sign(xwidth_mm) == 1
    xyz = estimate_probe_xyz_position(angle_downward_inclination,
                                      np.abs(xwidth_mm),
                                      probe_depth_mm,
                                      right_hemisphere=right_hemisphere,
                                      towards_midline=True,
                                      **kwargs)
    if dv_projection_mm and verbose is True:
        info = (dv_projection_mm, xyz[-1] - -dv_projection_mm)
        print('DV difference %0.04f[mm]: (diff=%0.04f[mm])'%info)
    if verbose is True:
        print(xyz)
    return xyz

def estimate_probe_xyz_position(angle_downward_inclination,
                                xwidth_mm,
                                probe_depth_mm,
                                right_hemisphere=True,
                                towards_midline=True,
                                angle_azimuth_nominal=45,
                                verbose=False,
                                ):
    '''All terms relative to manipulator position.

    Notes
    -----
    Convention:
        x: right(+)/left(-)
        y: anterior(+)
        z: dorsal(+)/ventral(-)

    Parameters
    ----------
    angle_downard_inclination : scalar, [deg]
         Angle of manipulator pointing down
    xwidth_mm : scalar, [mm]
        Extent of probe in horizontal axis (e.g. size on 2D coronal projection)
    probe_depth_mm : scalar, [mm]
        Size of probe inside of brain

    Returns
    -------
    xyz_coords : np.ndarray, (3,)
        Position of probe tip in cartesian coordinates.
        Because insertions typically pointing down, z is typically negative and -x is LH.
    '''
    # force downwardness
    angle_inclination = np.mod(angle_downward_inclination, 90) + 90

    if right_hemisphere:
        flip = -1 if towards_midline else 1
    else:
        flip = 1 if towards_midline else -1

    xpos = xwidth_mm*flip
    angle_azimuth = np.rad2deg(np.arccos(xpos/(
        probe_depth_mm*np.sin(np.deg2rad(angle_inclination)))))

    ypos = probe_depth_mm*(np.sin(np.deg2rad(angle_inclination)) *
                        np.sin(np.deg2rad(angle_azimuth)))

    xpos = probe_depth_mm*(np.sin(np.deg2rad(angle_inclination)) *
                        np.cos(np.deg2rad(angle_azimuth)))

    zpos = probe_depth_mm*np.cos(np.deg2rad(angle_inclination))

    radius = np.sqrt(np.sum(xpos**2 + ypos**2 + zpos**2))
    assert np.allclose(probe_depth_mm, radius)

    xyz_from_angles = estimate_probe_xyz_from_angles(angle_inclination,
                                                     90 - angle_azimuth_nominal*flip,
                                                     probe_depth_mm)
    xyz_from_proj = np.asarray([xpos, ypos, zpos])
    if verbose:
        print('Difference: from 90 angles (azimuth=%0.02f, incli=%0.02f):'%(90-angle_azimuth, 90-angle_inclination),
          xyz_from_proj - xyz_from_angles)
    return xyz_from_proj


def test_estimate_probe_xyz_position():
    # on the right side of the brain, pointing towards the left (midline)

    xyz_from_angles = estimate_probe_xyz_from_angles(30+90, 90+40, 3.84)

    xwidth = np.abs(xyz_from_angles[0])
    xyz_for_brain = estimate_probe_xyz_position(30 + 90,
                                                xwidth=xwidth,
                                                probe_depth=3.84,
                                                right_hemisphere=True,
                                                towards_midline=True)
    print(xyz_for_brain)
    assert np.allclose(xyz_from_angles, xyz_for_brain)

    # on the left side of the brain, pointing towards the right (midline)
    xyz_from_angles = estimate_probe_xyz_from_angles(30+90, 90-40, 3.84)
    xwidth = np.abs(xyz_from_angles[0])
    xyz_for_brain = estimate_probe_xyz_position(30 + 90,
                                                xwidth=xwidth,
                                                probe_depth=3.84,
                                                right_hemisphere=False,
                                                towards_midline=True)
    assert np.allclose(xyz_from_angles, xyz_for_brain)


    # on the right side of the brain, pointing towards the right (towards the outside)
    xyz_from_angles = estimate_probe_xyz_from_angles(30+90, 90-40, 3.84)
    xwidth = np.abs(xyz_from_angles[0])
    xyz_for_brain = estimate_probe_xyz_position(30 + 90,
                                                xwidth=xwidth,
                                                probe_depth=3.84,
                                                right_hemisphere=True,
                                                towards_midline=False)
    assert np.allclose(xyz_from_angles, xyz_for_brain)


    # on the left side of the brain, pointing towards the left (towards the outside)
    xyz_from_angles = estimate_probe_xyz_from_angles(30+90, 90+40, 3.84)
    xwidth = np.abs(xyz_from_angles[0])
    xyz_for_brain = estimate_probe_xyz_position(30 + 90,
                                                xwidth=xwidth,
                                                probe_depth=3.84,
                                                right_hemisphere=False,
                                                towards_midline=False)
    assert np.allclose(xyz_from_angles, xyz_for_brain)



def allenccf_main_areas(min_nparents=6, max_nparents=np.inf,
                        ignore=[695,315,997], verbose=False,
                        ):
    '''Get subset of Allen CCF areas that are parents (i.e. not VISp layer 1, etc).

    Returns
    -------
    main_areas : dict
        Allen CCF area annotations by index.
    '''
    import pandas
    from fusi.config import DATA_ROOT
    table = pandas.read_csv(f'{DATA_ROOT}/extras/structure_tree_safe_2017.csv')

    parents = {int(table.id[idx]) : '/'.join(t.split('/')[:-2]) for idx, t in enumerate(np.asarray(table.structure_id_path.values))}
    nparents = {int(table.id[idx]) : len(t.split('/')) for idx, t in enumerate(np.asarray(table.structure_id_path.values))}
    unique_parents = {int(t.split('/')[-1]) : t for t in np.unique([t for t in parents.values()]) if t != ''}
    unique_colors = np.unique(np.asarray(table.color_hex_triplet.values))

    main_areas = {}
    children2areas = {}

    for color in unique_colors:
        matched_color = table.id[table.color_hex_triplet==color]
        matched_nparents = np.asarray([nparents[ii] for ii in table.id[table.color_hex_triplet==color]])

        current_nparents = np.min(matched_nparents)
        ncandidates = (matched_nparents == current_nparents).sum()

        if (current_nparents < min_nparents) or (current_nparents > max_nparents):
            continue

        candidates = [t for t in matched_color if ((t in unique_parents) and (nparents[t]==current_nparents))]
        candidates = [t for t in candidates if t not in ignore]

        if len(candidates) == 0:
            continue

        for icand, cand in enumerate(candidates):
            area_id = matched_color[matched_color.index[icand]]
            area_idx = table[table.id==area_id].index[0]
            main_areas[area_idx] = color

            if verbose:
                print(area_id, area_idx,
                      str(table[table.id==area_id].acronym.values[0]),
                      str(table[table.id==area_id].name.values[0]),
                      color, nparents[area_id])

    return main_areas


def main_area_contours():
    pass



def cleanup_vasculature_atlas(arr, anterior=False):
    '''Fix incorrect values in vasculature atlas.

    For whatever reason, the VesSAP Allen CCF atlas has codes
    that are missing in the Allen CCF tables. This function
    replaces those codes with the correct ones.

    The code for optic radiation is wrong, it maps to 484682496 Prosubiculum.

    NB: When given a correct atlas, this function should

    Parameters
    ----------
    arr : np.ndarray, or nibabel.Nifti1Image
        VesSAP vasculature atlas volume whose entries are Allen CCF IDs
    anterior : bool
        If True, change the code for the optic radiation

    Returns
    -------
    out : np.ndarray, or nibabel.Nifti1Image
    '''
    import pandas
    from fusi.config import DATA_ROOT
    csvpath = f'{DATA_ROOT}/extras/structure_tree_safe_2017.csv'

    is_image = False
    if isinstance(arr, nib.Nifti1Image):
        print('Loading data from nifti...')
        aff = arr.affine.copy()
        arr = np.asarray(arr.get_fdata()).copy()
        is_image = True

    allen_ids = np.asarray(pandas.read_csv(csvpath).id.values).astype(np.int)
    vasc_ids = np.unique(arr).astype(np.int)

    vasc_matches = np.in1d(vasc_ids, allen_ids)
    vasc_missing = vasc_ids[np.logical_not(vasc_matches)]

    atlas_not_in_vasc = np.unique(allen_ids[np.logical_not(np.in1d(allen_ids, vasc_ids))])


    if vasc_missing[0] == 0:
        # drop zero
        vasc_missing = vasc_missing[1:]

    # Where possible, incorrect value replaced by the layer value so that
    # we can find its parent area  the same way as correct values.
    # Otherwise it's the closest value
    value = 0 #
    fixes = {
        997 : 0,                # Do not show root node as white
        32768 : 0, # This is the edges of the affine transform from VesSAP
        182305696 : 182305693, # Primary somatosensory area unassigned layer 1
        182305712 : 182305693, # Primary somatosensory area unassigned layer 1
        312782560 : 312782550, # Anterior area layer 1
        312782592 : 312782578, # Laterointermediate area layer 1
        312782656 : 312782632, # Postrhinal area layer 1
        484682464 : 484682470, # Prosubiculum
        526157184 : 526157196, # Frontal pole layer 6a (closest actual match)
        526322272 : 526322264, # Frontal pole layer 6b (closest actual match)
        527696992 : 527696977, # Orbital area medial part layer 6b (closets actual match)
        549009216 : 549009211, # Medial accesory oculomotor nucleus but also somewhere quite posterior.... (Peritrigeminal zone)
        560581568 : 560581551, # Ethmoid nucleus of the thalamus (good match),
        563807424 : 560581563, # This label is for 2 areas: Posterior intralaminar thalamic nucleus and (563807435) Posterior triangular thalamic nucleus
        576073728 : 576073704, # Perifornical nucleus
        589508416 : value,
        589508480 : 589508451,  # Paratrigeminal nucleus
        599626944 : 599626923,  # subcommissural organ
        606826624 : 606826647, # Medial mammillary nucleus lateral part
        606826688 : 606826663, # Paratrochlear nucleus
        607344832 : 607344838, # Interpeduncular nucleus caudal (lateral, etc, not in vasc.)
        614454272 : 614454277, # Supraoculomotor periaqueductal gray
    }

    if anterior:
        # this code corresponds to two areas in the vasc atlas.
        # it is correct in the back of the brain wher eit is the postsubiculum or area prostriata
        # oddly, in the frontal regions the optic radiation is marked by this code.
        # this replaces that code to be the "optic radiation" which is colored in grey
        fixes[484682496] = 484682520,

    for idx, (value_wrong, value_correct) in enumerate(fixes.items()):
        missing = arr==value_wrong
        print('Working on %i/%i: %i -> %i (#%i voxels)'%(idx+1,len(fixes),
                                                         value_wrong, value_correct,
                                                         missing.sum()))
        arr[missing] = value_correct

    if is_image:
        arr = nib.Nifti1Image(arr, affine=aff)

    return arr


def remap_allen_id2index(dat):
    '''Convert Allen CCF ID codes to their index on the CSV table

    Allen CCF IDs are cumbersome. This function remaps them to Cortex Lab convention.

    Parameters
    ----------
    dat : nibabel.Nifti1Image
        Allen CCF volume whose entries are area IDs

    Returns
    im : nibabel.Nifti1Image
        Volume whose entries are the table indeces of the Allen CCF IDs.
    '''
    from fusi.config import DATA_ROOT
    csvpath = f'{DATA_ROOT}/extras/structure_tree_safe_2017.csv'

    table = np.loadtxt(csvpath, delimiter=',', dtype='S')
    content = table[1:]
    column_names = table[0, :]
    print(column_names)
    data_id = list(content[:,0])
    data_text = list(content[:,column_names == b'safe_name'].squeeze())
    data_text = ['_'.join(words.decode().split()) for words in data_text]
    data_hexcolors = content[:,column_names == b'color_hex_triplet'].squeeze()
    data_colors = list(map(hex2rgba, data_hexcolors))


    arr = np.asarray(dat.get_fdata().copy()).astype(np.int)
    valid_ids = np.unique(arr.ravel()).astype(np.int)
    narr = np.zeros_like(arr)

    found = 0
    for rowidx, (idd, name, rgba) in enumerate(zip(data_id,
                                                   data_text,
                                                   data_colors)):
        # if rowidx ==0:
        #     continue
        # if name == '' or
        if idd.decode() == '':
            continue

        idd = int(idd.decode())
        if (valid_ids == idd).sum() == 0:
            continue

        found += 1
        print('Working on ID=%i row=%i'%(idd, rowidx), arr.dtype, narr.dtype, found)
        matches = arr==idd
        if matches.sum() > 0:
            narr[matches] = rowidx
    nim = nib.Nifti1Image(narr, affine=dat.affine, header=dat.header)
    return nim

def allenccf_cmap(matplotlib=True):
    import matplotlib.colors
    import pandas
    from fusi.config import DATA_ROOT

    csvpath = f'{DATA_ROOT}/extras/structure_tree_safe_2017.csv'
    table = pandas.read_csv(csvpath)

    colors = {}
    cmap = []
    for idx, color in enumerate(table.color_hex_triplet):
        colors[idx] = hex2rgba(color)
        hexcolor = '#%s'%color
        cmap.append(hexcolor)
    cmap = matplotlib.colors.ListedColormap(cmap, name='allenccf')
    norm = matplotlib.colors.BoundaryNorm(np.arange(table.shape[0]), table.shape[0]-1, clip=True)
    return cmap, norm

def fusiarr2nii(arr, xyzmm=(0.1, 0.1, 0.04832), flips=(1, 1, -1)):
    '''Convert a fUSi array to a RAS nifti image

    Parameters
    ----------
    arr : np.ndarray, (z,x,y)
        z: superior->inferior
        x: left->right
        y: posterior->anterior

    Returns
    -------
    nifti : nibabel.Nifti1Image
        x: R->L # sagital slices
        y: A->P # coronal slices
        z: S->I # axial slices
    '''
    xyzmm = np.asarray(xyzmm)
    flips = np.asarray(flips)

    # orient the image to: A/P S/I R/L
    swaps = (1, 2, 0)
    arr = arr.transpose(swaps)

    swaps = np.asarray(swaps)
    dimensions = xyzmm
    im = array2nifti(arr, dimensions,
                     # dimensions[swaps],
                     flips=np.asarray(flips)) # flip last dimension
    print(arr.shape)
    return im


def fusi2nii(fusi_matfile, key=None, normalize=True, datakey='YStack'):
    '''
    '''
    from scipy import io as sio

    matdat = sio.loadmat(fusi_matfile,
                         struct_as_record=False,
                         squeeze_me=True)

    # Check contents of MATLAB file are as expected
    keys = [k for k in list(matdat.keys()) if '__' not in k]
    key = 'ys'
    assert key in keys

    data = matdat[key]
    contents = data.__dict__.keys()
    assert 'xAxis' in contents
    assert 'yAxis' in contents
    assert 'zAxis' in contents
    # assert 'YStack' in contents
    assert datakey in contents

    # report
    print('Working on: %s (%s)'%(matdat['animalName'], getattr(data, 'ExpRef')))

    # Load data
    arr = getattr(data, datakey)
    arr = np.asarray(arr, dtype=np.float32).T
    xsamples = getattr(data, 'xAxis')
    ysamples = getattr(data, 'yAxis')
    zsamples = getattr(data, 'zAxis')

    if normalize:
        arr -= arr.min()
        arr /= arr.max()
        arr *= 1000

    # find size in mm
    xmm = np.unique(np.diff(xsamples))[0]
    ymm = np.unique(np.diff(ysamples))[0]
    zmm = np.unique(np.diff(zsamples))[0]
    dimensions = np.asarray([xmm, ymm, zmm])

    # orient the image to: A/P S/I R/L
    swaps = (1, 0, 2)
    arr = arr.transpose(swaps)

    im = array2nifti(arr, dimensions[np.asarray(swaps)],
                     flips=(1.0, 1.0, -1.0)) # flip last dimension
    print(arr.shape)
    return im


def old_fusi2nii(fusi_matfile, key=None, normalize=False):
    '''
    '''
    from scipy import io as sio

    dat = sio.loadmat(fusi_matfile)
    if key is None:
        contents = [k for k in list(dat.keys()) if '__' not in k]
        assert len(contents) == 1
        key = contents[0]

    arr = np.asarray(dat[key][0,0][0]).astype(np.float32).T
    if normalize:
        arr -= arr.min()
        arr /= arr.max()
        arr *= 1000

    # sample points in [mm]
    xsamples = dat[key][0,0][2]
    ysamples = dat[key][0,0][1]
    zsamples = dat[key][0,0][3]

    # find size in mm
    xmm = np.unique(np.diff(xsamples))[0]
    ymm = np.unique(np.diff(ysamples))[0]
    zmm = np.unique(np.diff(zsamples))[0]
    dimensions = np.asarray([xmm, ymm, zmm])

    # orient the image to: A/P S/I R/L
    swaps = (1, 0, 2)
    arr = arr.transpose(swaps)

    # TODO, BF: THIS IS PROBABLY WRONG.
    # THE DIMENSIONS ARE THE SAME AND DO NOT NEED SWAPING
    im = array2nifti(arr, dimensions[np.asarray(swaps)],
                     flips=(1.0, 1.0, -1.0)) # flip last dimension
    print(arr.shape)
    return im


def get_center_point(array_shape):
    '''Find the center of an array

    Parameters
    ----------
    array_shape : list-like
        Shape of the array

    Returns
    -------
    center : array
        The mid-point of the array
    '''
    center = (np.array(array_shape) - 1) / 2.
    return center


def array2nifti(arr, xyzmm, flips=(1, 1, 1)):
    '''Make a isocenter nifti image from an array

    Parameters
    ----------
    arr : 3D np.ndarray
        Image data
    xyzmm : list-like
        Size of voxels in milimeters
    flips : list-like, optional
        Mirror a given axis

    Returns
    -------
    image : nibabel.Nifti1Image
        Image with an iso-centered affine
        such that the voxel in the middle of the image
        is at coordinates (0mm, 0mm, 0mm).

        x: R->L # sagital slices (0:ventral->dorsal, 1:posterior->anterior)
        y: A->P # coronal slices (0:ventral->dorsal, 1:left->right)
        z: S->I # axial slices (0:posterior->anterior 1:left->right)
    '''
    # make affine
    assert np.allclose(np.abs(flips), 1)
    flipped_xyz = np.asarray(xyzmm)*np.asarray(flips)
    dimensions = np.hstack([flipped_xyz, [1]])
    affine = np.diag(dimensions)

    # move center of image to iso-center
    center_ijk = get_center_point(arr.shape)
    center_xyz = nib.affines.apply_affine(affine, center_ijk)
    affine[:-1, -1] = -center_xyz

    new_center = nib.affines.apply_affine(affine, center_ijk)
    assert np.allclose(new_center, 0)
    header = nib.Nifti1Header()
    header.set_qform(affine, code='scanner')
    header.set_sform(affine, code='scanner')
    image = nib.Nifti1Image(arr, affine, header=header)
    print(image.affine)
    return image


def make_nifti(arr, affine):
    '''Create a nifti image with the given affine

    Parameters
    ----------
    arr : 3D np.ndarray
        Image data
    affine : 2D np.ndarray, (4,4)
        Grid-to-world transformation

    Returns
    -------
    nifti_image : nibabel.Nifti1Image
        nibabel image with qform and sform code `scanner`.
    '''
    header = nib.Nifti1Header()
    header.set_qform(affine, code='scanner')
    header.set_sform(affine, code='scanner')
    return nib.Nifti1Image(arr, affine, header=header)


def estimate_affine_alignment(source_data,
                              dest_data,
                              source_affine=np.eye(4),
                              dest_affine=np.eye(4),
                              level_iters=[100, 20, 10],
                              sigmas=[3.0, 1.0, 0.0],
                              factors=[4, 2, 1]):
    """Perform a step-wise affine transformation.

    First, the images' center of mass is found. Then,
    the optimal translation is found and given as a
    starting affine for a rigid-body transform. Finally,
    a full affine search is conducted with the rigid-body
    transform as its initialization.

    Uses DiPy's affine transformation code.

    Parameters
    ----------
    dest_data : 3D np.ndarray
        destination/static reference image
    source_data : 3D np.ndarray
        source/moving image
    dest_affine, source_affine: 2D np.ndarray
        Grid-to-world  affine matrix

    Returns
    -------
    affine_transforms : dict
        Contains the transform objects for
        each step

        Contents:
        * `translation` transform:  3 DoF
        * `rigid` transform:        6 DoF
        * `affine` transform:      12 DoF

    Notes
    -----
    This is based on the great DiPy example:
    """
    from dipy.align.imaffine import (transform_centers_of_mass,
                                     MutualInformationMetric,
                                     AffineRegistration)
    from dipy.align.transforms import (TranslationTransform3D,
                                       RigidTransform3D,
                                       AffineTransform3D)
    from dipy.align.metrics import CCMetric

    from fusi.utils import StopWatch

    print('Mapping from', source_data.shape,
          'to', dest_data.shape)


    start = time.time()
    chronometer = StopWatch(verbose=True)
    results = {}

    dest = dest_data
    dest_affine = dest_affine

    source = source_data
    source_affine = source_affine

    # start by aligning to center of mass
    print('Working on: center of mass')
    c_of_mass = transform_centers_of_mass(dest, dest_affine,
                                          source, source_affine)
    chronometer()
    print(c_of_mass.affine)

    # setup the MI cost function
    ##############################
    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)

    # setup affine registartion
    ##############################
    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors,
                                verbosity=3)

    print('Working on: translation-only (3 DoF)')
    transform = TranslationTransform3D()
    params0 = None
    starting_affine = c_of_mass.affine
    translation = affreg.optimize(dest, source, transform, params0,
                                  dest_affine, source_affine,
                                  starting_affine=starting_affine)
    results['translation'] = translation
    chronometer()
    print(translation.affine)

    print('Working on: rigid transform-only (6 DoF)')
    transform = RigidTransform3D()
    params0 = None
    starting_affine = translation.affine
    rigid = affreg.optimize(dest, source, transform, params0,
                            dest_affine, source_affine,
                            starting_affine=starting_affine)
    results['rigid'] = rigid
    chronometer()
    print(rigid.affine)


    print('Working on: affine transform (12 DoF)')
    transform = AffineTransform3D()
    params0 = None
    starting_affine = rigid.affine
    affine = affreg.optimize(dest, source, transform, params0,
                             dest_affine, source_affine,
                             starting_affine=starting_affine)
    results['affine'] = affine
    chronometer()
    print(affine.affine)

    print('Total duration: %0.2f[sec]'%(time.time() - start))
    return results


def test_array2nifti():
    arr = np.random.randn(10, 11, 12)
    xyzmm = [0.1, 0.2, 0.3]
    image = array2nifti(arr, xyzmm)
    center_xyz = nib.affines.apply_affine(image.affine, get_center_point(arr.shape))
    assert np.allclose(center_xyz, 0)
    assert image.header.structarr['sform_code'] == 1
    assert image.header.structarr['qform_code'] == 1


if __name__ == '__main__':
    pass
