import numpy as np
import pandas as pd
from scipy import ndimage

"""These functions are taken from Alex Gonzalez's TreeMazeAnalyses2 package:
https://github.com/alexgonzl/TreeMazeAnalyses2/blob/e4f92406b5e3b5281c5d092515ee10093be216ea/Analyses/spatial_functions.py"""

def compute_border_score_solstad(fr_maps, fr_thr=0.25, min_field_size_bins=10, width_bins=3, return_all=False):
    """
    Border score method from Solstad et al Science 2008. Returns the border score along with the max coverage by a field
    and the weighted firing rate. This works for a single fr_map or multiple.
    :param fr_maps: np.ndarray, (dimensions can be 2 or 3), if 3 dimensions, first dimensions must
                    correspond to the # of units, other 2 dims are height and width of the map
    :param fr_thr: float, proportion of the max firing rate to threshold the data
    :param min_field_size_bins: int, # of bins that correspond to the total area of the field. fields found
                    under this threshold are discarded
    :param width_bins: wall width by which the coverage is determined.
    :param return_all: bool, if False only returns the border_score
    :return: border score, max coverage, distanced weighted neural_data for each unit in maps.
    -> code based of the description on Solstad et al, Science 2008
    Default kwargs set from: https://github.com/alexgonzl/TreeMazeAnalyses2/blob/e4f92406b5e3b5281c5d092515ee10093be216ea/Analyses/open_field_functions.py#L1360-L1362
    """
    n_walls = 4
    # add a singleton dimension in case of only one map to find fields.
    if fr_maps.ndim == 2:
        fr_maps = fr_maps[np.newaxis,]
    n_units, map_height, map_width = fr_maps.shape

    # get fields
    field_maps, n_fields = get_map_fields(fr_maps, thr=fr_thr, min_field_size=min_field_size_bins)

    if field_maps.ndim == 2:
        field_maps = field_maps[np.newaxis,]
        n_fields = n_fields[np.newaxis,]

    # get border distance matrix
    distance_mat = get_center_border_distance_mat(map_height, map_width)  # linear distance to closest wall [bins]

    # get wall labels
    wall_labels_mask = get_wall_masks(map_height, map_width, width_bins)

    # pre-allocate scores
    border_score = np.zeros(n_units) * np.nan
    border_max_cov = np.zeros(n_units) * np.nan
    border_w_fr = np.zeros(n_units) * np.nan

    def _border_score_solstad(_field_map, _fr_map, _distance_mat, _wall_labels_mask):
        """
        computes the border scores given the field id map, firing rate and wall_mask
        :param _fr_map: 2d firing rate map
        :param _field_map: as obtained from get_map_fields
        :param _wall_labels_mask: as obtained from get_wall_masks
        :return: border_score, max_coverage, weighted_fr
        """
        _n_fields = int(np.max(_field_map)) + 1

        wall_coverage = np.zeros((_n_fields, n_walls))
        for field in range(_n_fields):
            for wall in range(n_walls):
                wall_coverage[field, wall] = np.sum(
                    (_field_map == field) * (_wall_labels_mask[wall] == wall)) / np.sum(
                    _wall_labels_mask[wall] == wall)
        c_m = np.max(wall_coverage)

        # get normalized distanced weighted firing rate
        field_fr_map = _fr_map * (_field_map >= 0)
        d_m = np.sum(field_fr_map * _distance_mat) / np.sum(field_fr_map)

        # get border score
        b = (c_m - d_m) / (c_m + d_m)
        return b, c_m, d_m

    # loop and get scores
    for unit in range(n_units):
        fr_map = fr_maps[unit]
        field_map = field_maps[unit]
        n_fields_unit = n_fields[unit]
        if n_fields_unit > 0:
            border_score[unit], border_max_cov[unit], border_w_fr[unit] = \
                _border_score_solstad(field_map, fr_map, distance_mat, wall_labels_mask)

    if return_all:
        return border_score, border_max_cov, border_w_fr
    else:
        return border_score

# -border aux
def get_center_border_distance_mat(h, w):
    """
    creates a pyramid like matrix of distances to border walls.
    :param h: height
    :param w: width
    :return: normalized matrix of distances, center =1, borders=0
    """
    a = np.arange(h)
    b = np.arange(w)

    r_h = np.minimum(a, a[::-1])
    r_w = np.minimum(b, b[::-1])
    pyr = np.minimum.outer(r_h, r_w)
    return pyr / np.max(pyr)


def get_wall_masks(map_height, map_width, wall_width):
    """
    returns a mask for each wall. *assumes [0,0] is on lower left corner.*
    :param map_height:
    :param map_width:
    :param wall_width: size of the border wall
    :return: mask, ndarray size 4 x map_height x map_width, 4 maps each containing a mask for each wall
    """

    mask = np.ones((4, map_height, map_width), dtype=int) * -1

    mask[0][:, map_width:(map_width - wall_width - 1):-1] = 0  # right / East
    mask[1][map_height:(map_height - wall_width - 1):-1, :] = 1  # top / north
    mask[2][:, 0:wall_width] = 2  # left / West
    mask[3][0:wall_width, :] = 3  # bottom / south

    return mask

def get_map_fields(maps, thr=0.3, min_field_size=20, filt_structure=None):
    """
    gets labeled firing rate maps. works on either single maps or an array of maps.
    returns an array of the same dimensions as fr_maps with
    :param maps: np.ndarray, (dimensions can be 2 or 3), if 3 dimensions, first dimensions must
                    correspond to the # of units, other 2 dims are height and width of the map
    :param thr: float, proportion of the max firing rate to threshold the data
    :param min_field_size: int, # of bins that correspond to the total area of the field. fields found
                    under this threshold are discarded
    :param filt_structure: 3x3 array of connectivity. see ndimage for details
    :return field_labels (same dimensions as input), -1 values are background, each field has an int label
    """
    if filt_structure is None:
        filt_structure = np.ones((3, 3))

    # add a singleton dimension in case of only one map to find fields.
    if maps.ndim == 2:
        maps = maps[np.newaxis, :, :]
    elif maps.ndim == 1:
        print('maps is a one dimensional variable.')
        return None

    n_units, map_height, map_width = maps.shape

    # create border mask to avoid elimating samples during the image processing step
    border_mask = np.ones((map_height, map_width), dtype=bool)
    border_mask[[0, -1], :] = False
    border_mask[:, [0, -1]] = False

    # determine thresholds
    max_fr = maps.max(axis=1).max(axis=1)

    # get fields
    field_maps = np.zeros_like(maps)
    n_fields = np.zeros(n_units, dtype=int)
    for unit in range(n_units):
        # threshold the maps
        thr_map = maps[unit] >= max_fr[unit] * thr

        # eliminates small/noisy fields, fills in gaps
        thr_map = ndimage.binary_closing(thr_map, structure=filt_structure, mask=border_mask)
        thr_map = ndimage.binary_dilation(thr_map, structure=filt_structure)

        # get fields ids
        field_map, n_fields_unit = ndimage.label(thr_map, structure=filt_structure)

        # get the area of the fields in bins
        field_sizes = np.zeros(n_fields_unit)
        for f in range(n_fields_unit):
            field_sizes[f] = np.sum(field_map == f)

        # check for small fields and re-do field identification if necessary
        if np.any(field_sizes < min_field_size):
            small_fields = np.where(field_sizes < min_field_size)[0]
            for f in small_fields:
                thr_map[field_map == f] = 0
            field_map, n_fields_unit = ndimage.label(thr_map, structure=filt_structure)

        # store
        field_maps[unit] = field_map
        n_fields[unit] = n_fields_unit

    field_maps -= 1  # make background -1, labels start at zero

    # if only one unit, squeeze to match input dimensions
    if n_units == 1:
        field_maps = field_maps.squeeze()
        n_fields = n_fields.squeeze()

    return field_maps, n_fields
