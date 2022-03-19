import numpy as np
import os, copy
import scipy
import scipy.io as spio
from scipy.stats import binned_statistic, binned_statistic_2d
import xarray as xr
import pickle
from mec_hpc_investigations.core.constants import CAITLIN1D_VR_SESSIONS, gridscore_starts, gridscore_ends
from mec_hpc_investigations.core.default_dirs import BASE_DIR_RESULTS, CAITLIN1D_VR_PACKAGED, CAITLIN1D_VR_TRIALBATCH_DIR
from mec_hpc_investigations.models.scores import GridScorer
from mec_hpc_investigations.neural_fits.utils import package_scores
from mec_hpc_investigations.neural_data.border_score_utils import compute_border_score_solstad
from mec_hpc_investigations.neural_data.head_direction_score_utils import resultant_vector_length

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def extract_session_tetrode_id(cell_id, animal_id):
    '''
    Given: cell_id and animal_id
    Returns: session_id and tetrode_id extracted from the cell_id
    '''
    tetrode_id = cell_id.split('_')[-1]
    session_idx_start = len(animal_id) + 1 # include underscore
    session_idx_end = len(cell_id) - (len(tetrode_id) + 1) # include underscore
    session_id = cell_id[session_idx_start:session_idx_end]
    # sanity check
    constructed_cell_id = animal_id + '_' + session_id + '_' + tetrode_id
    assert(constructed_cell_id == cell_id)
    return session_id, tetrode_id

def populate_session_tetrode_id(d):
    '''
    Given: dictionary d of all attributes, adds session_id and tetrode_id
    derived from cell_id
    '''
    session_id, tetrode_id = extract_session_tetrode_id(cell_id=d["cell_id"],
                                                       animal_id=d["animal_id"])
    d["session_id"] = session_id
    d["tetrode_id"] = tetrode_id
    return d


def get_xy_bins(curr_arena_dataset, nbins_max=20):
    # Compute min/max position across all animals in that arena/environment
    # These are the bins to be used for all animals within that arena.
    body_x_min = np.amin([np.nanmin(body_val) for body_val in curr_arena_dataset["body_position_x"]])
    body_x_max = np.amax([np.nanmax(body_val) for body_val in curr_arena_dataset["body_position_x"]])

    body_y_min = np.amin([np.nanmin(body_val) for body_val in curr_arena_dataset["body_position_y"]])
    body_y_max = np.amax([np.nanmax(body_val) for body_val in curr_arena_dataset["body_position_y"]])

    # Bin, strategy adopted from Mallory, Hardcastle et al. 2021
    nbins = (int)(curr_arena_dataset["arena_size_cm"][0]*(nbins_max/100.0))
    arena_x_bins = np.linspace(body_x_min, body_x_max, nbins+1, endpoint=True)
    arena_y_bins = np.linspace(body_y_min, body_y_max, nbins+1, endpoint=True)
    return arena_x_bins, arena_y_bins

def compute_binned_frs(cell_idx,
                       curr_arena_dataset,
                       arena_x_bins,
                       arena_y_bins,
                       smooth_std=1,
                       dt=0.02,
                       verbose_return=False,
                       shift=False,
                       return_unbinned=False):
    """
    Computes rate maps from data.
    """
    t = curr_arena_dataset["time"][cell_idx]
    diff_t = np.diff(t)
    # sometimes this can be 0.019999...
    assert(np.isclose(diff_t, dt*np.ones_like(diff_t)).all())
    spt = curr_arena_dataset["spike_times"][cell_idx]
    curr_body_x = curr_arena_dataset["body_position_x"][cell_idx]
    curr_body_y = curr_arena_dataset["body_position_y"][cell_idx]
    # find the index in t of the time a spike occured
    t_indices = np.where(np.in1d(t, spt))[0]
    # find the counts of each spike time
    _, c = np.unique(spt, return_counts=True)
    sp_counts = np.zeros(t.shape)
    # associate the times with the counts
    sp_counts[t_indices] = c
    assert(np.sum(sp_counts) == spt.shape[0])
    # for permutation testing, to allow for shifting the spike counts per cell
    if shift:
        n_samps = len(curr_body_x)
        assert n_samps == sp_counts.shape[0], 'Mismatch lengths between samples and neural_data.'
        sp_counts = np.roll(sp_counts, shift=np.random.randint(n_samps))

    if return_unbinned:
        sp_counts = sp_counts * (1.0/dt)
        return sp_counts

    ret_val = {}
    if verbose_return:
        # Compute rate maps
        curr_binned_pos_counts = binned_statistic_2d(x=curr_body_x,
                                              y=curr_body_y,
                                              values=None,
                                              statistic="count",
                                              bins=[arena_x_bins, arena_y_bins],
                                              expand_binnumbers=True)
        curr_binned_pos_counts = curr_binned_pos_counts.statistic
        ret_val["pos_counts"] = curr_binned_pos_counts

        curr_binned_spike_counts = binned_statistic_2d(x=curr_body_x,
                                              y=curr_body_y,
                                              values=sp_counts,
                                              statistic="sum",
                                              bins=[arena_x_bins, arena_y_bins],
                                              expand_binnumbers=True)
        curr_binned_spike_counts = curr_binned_spike_counts.statistic
        ret_val["spike_counts"] = curr_binned_spike_counts

    # this is equivalent to summing the sp_counts above for values in each bin
    # then dividing by the counts in each position bin * (dt, which is the time it takes in each position)
    curr_binned_frs = binned_statistic_2d(x=curr_body_x,
                                          y=curr_body_y,
                                          values=sp_counts,
                                          statistic="mean",
                                          bins=[arena_x_bins, arena_y_bins],
                                          expand_binnumbers=True)
    curr_binned_avg_frs = curr_binned_frs.statistic*(1.0/dt)
    if verbose_return:
        ret_val["frs_raw"] = curr_binned_avg_frs

    if smooth_std is not None:
        curr_binned_avg_frs = scipy.ndimage.gaussian_filter(input=np.nan_to_num(curr_binned_avg_frs),
                                                             sigma=smooth_std, mode="constant")

        if verbose_return:
            ret_val["smoothed_frs"] = curr_binned_avg_frs


    if verbose_return:
        return ret_val
    else:
        return curr_binned_avg_frs

def aggregate_responses(dataset,
                        arena_sizes=None,
                        nbins_max=20,
                        **kwargs):

    """
    Aggregates responses across cells in a given animal, per arena.
    """
    if arena_sizes is None:
        arena_sizes = list(np.unique(dataset["arena_size_cm"]))
    else:
        if not isinstance(arena_sizes, list):
            arena_sizes = [arena_sizes]

    spec_resp_agg = {}
    for arena_size in arena_sizes:
        spec_resp_agg[arena_size] = {}
        curr_arena_dataset = dataset[dataset["arena_size_cm"] == arena_size]
        arena_x_bins, arena_y_bins = get_xy_bins(curr_arena_dataset, nbins_max=nbins_max)

        for animal in np.unique(curr_arena_dataset["animal_id"]):
            curr_animal_resp = []
            cell_ids = []
            curr_animal_arena_dataset = curr_arena_dataset[curr_arena_dataset["animal_id"] == animal]
            for cell_idx in range(curr_animal_arena_dataset.shape[0]):
                cell_ids.append(curr_animal_arena_dataset[cell_idx]["cell_id"])
                # note that even cells within an animal can have different x and y positions (since they can be from different recording sessions), so we bin per cell separately
                ret_val = compute_binned_frs(cell_idx=cell_idx,
                                               curr_arena_dataset=curr_animal_arena_dataset,
                                               arena_x_bins=arena_x_bins,
                                               arena_y_bins=arena_y_bins,
                                               **kwargs)
                curr_animal_resp.append(ret_val)

            spec_resp_agg[arena_size][animal] = {"resp": np.stack(curr_animal_resp, axis=-1),
                                                 "cell_ids": np.array(cell_ids)}

    return spec_resp_agg

def concat_resp_conds(dataset_cond_1, dataset_cond_2):
    """Concatenates the responses of the SAME population of cells across two conditions (e.g. Open Field & Task)."""
    concat_spec_resp_agg = {}
    for arena_size in dataset_cond_1.spec_resp_agg.keys():
        concat_spec_resp_agg[arena_size] = {}
        for animal in dataset_cond_1.spec_resp_agg[arena_size].keys():
            concat_spec_resp_agg[arena_size][animal] = {"resp": {}, "cell_ids": {}}
            dataset_cond1_cell_ids = dataset_cond_1.spec_resp_agg[arena_size][animal]["cell_ids"]
            dataset_cond2_cell_ids = dataset_cond_2.spec_resp_agg[arena_size][animal]["cell_ids"]
            assert(np.array_equal(dataset_cond1_cell_ids, dataset_cond2_cell_ids))
            concat_spec_resp_agg[arena_size][animal]["cell_ids"] = dataset_cond1_cell_ids
            dataset_cond1_resp = dataset_cond_1.spec_resp_agg[arena_size][animal]["resp"]
            dataset_cond2_resp = dataset_cond_2.spec_resp_agg[arena_size][animal]["resp"]
            assert(np.array_equal(dataset_cond1_resp.shape, dataset_cond2_resp.shape))
            num_cells = dataset_cond1_resp.shape[-1]
            # flatten and concatenate along stimuli dimension
            concat_resp = np.concatenate([dataset_cond1_resp.reshape((-1, num_cells)),
                                          dataset_cond2_resp.reshape((-1, num_cells))], axis=0)
            concat_spec_resp_agg[arena_size][animal]["resp"] = concat_resp
    return concat_spec_resp_agg


def get_coords(dataset, arena_size):
    """Get x,y coordinates for each position bin, in order to compute EMD distance M matrix."""
    arena_x_bins, arena_y_bins = get_xy_bins(dataset[dataset["arena_size_cm"] == arena_size])
    # ensure arena is evenly spaced
    assert(np.isclose(np.diff(arena_x_bins), np.diff(arena_x_bins)[0]).all())
    assert(np.isclose(np.diff(arena_y_bins), np.diff(arena_y_bins)[0]).all())
    # have coordinates be midpoints of states
    pix1 = (arena_x_bins + (np.diff(arena_x_bins)/2.0)[0])[:-1]
    pix2 = (arena_y_bins + (np.diff(arena_y_bins)/2.0)[0])[:-1]
    X, Y = np.meshgrid(pix1, pix2)
    coords = np.stack([X.flatten(), Y.flatten()], axis=-1)
    return coords

def aggregate_gridscore(dataset,
                        arena_sizes=None,
                        nbins_max=20,
                        min_max=True,
                        spec_resp_agg=None,
                        **kwargs):

    """
    Aggregates grid scores across cells in a given animal, per arena.
    Grid cells are cells with grid score > 0.3.
    """

    if spec_resp_agg is None:
        if hasattr(dataset, 'spec_resp_agg'):
            print("Using native spec resp agg")
            spec_resp_agg = dataset.spec_resp_agg

    if arena_sizes is None:
        if spec_resp_agg is not None:
            arena_sizes = list(spec_resp_agg.keys())
        else:
            arena_sizes = list(np.unique(dataset["arena_size_cm"]))
    else:
        if not isinstance(arena_sizes, list):
            arena_sizes = [arena_sizes]

    scores_agg = {}
    for arena_size in arena_sizes:
        scores_agg[arena_size] = {}
        if hasattr(dataset, "arena_x_bins"):
            assert(len(arena_sizes) == 1)
            arena_x_bins = dataset.arena_x_bins
        elif hasattr(dataset, "get_arena_bins"):
            arena_x_bins, arena_y_bins = dataset.get_arena_bins(arena_size)
        else:
            curr_arena_dataset = dataset[dataset["arena_size_cm"] == arena_size]
            arena_x_bins, arena_y_bins = get_xy_bins(curr_arena_dataset, nbins_max=nbins_max)

        scorer = GridScorer(nbins=len(arena_x_bins)-1,
                   mask_parameters=zip(gridscore_starts, gridscore_ends.tolist()),
                   min_max=min_max)

        if spec_resp_agg is not None:
            for animal in spec_resp_agg[arena_size].keys():
                curr_resp = spec_resp_agg[arena_size][animal]["resp"]
                curr_cell_ids = spec_resp_agg[arena_size][animal]["cell_ids"]
                curr_animal_scores = []
                for cell_idx in range(len(curr_cell_ids)):
                    ret_val = curr_resp[:, :, cell_idx]
                    curr_grid_score = scorer.get_scores(ret_val)[0]
                    curr_animal_scores.append(curr_grid_score)

                scores_agg[arena_size][animal] = package_scores(scores=np.stack(curr_animal_scores, axis=-1),
                                                                cell_ids=curr_cell_ids)

        else:
            for animal in np.unique(curr_arena_dataset["animal_id"]):
                curr_animal_scores = []
                cell_ids = []
                curr_animal_arena_dataset = curr_arena_dataset[curr_arena_dataset["animal_id"] == animal]
                for cell_idx in range(curr_animal_arena_dataset.shape[0]):
                    cell_ids.append(curr_animal_arena_dataset[cell_idx]["cell_id"])
                    # note that even cells within an animal can have different x and y positions, so we bin per cell separately
                    ret_val = compute_binned_frs(cell_idx=cell_idx,
                                                   curr_arena_dataset=curr_animal_arena_dataset,
                                                   arena_x_bins=arena_x_bins,
                                                   arena_y_bins=arena_y_bins,
                                                   **kwargs)
                    curr_grid_score = scorer.get_scores(ret_val)[0]
                    curr_animal_scores.append(curr_grid_score)

                scores_agg[arena_size][animal] = package_scores(scores=np.stack(curr_animal_scores, axis=-1),
                                                                cell_ids=np.array(cell_ids))

    return scores_agg

def aggregate_borderscore(dataset,
                        arena_sizes=None,
                        min_max=True,
                        spec_resp_agg=None,
                        border_score_params={},
                        **kwargs):

    """
    Aggregates border scores across cells in a given animal, per arena.
    Border cells are cells with border score > 0.5, as set by Soldstad et al. 2008.
    """

    if spec_resp_agg is None:
        if hasattr(dataset, 'spec_resp_agg'):
            print("Using native spec resp agg")
            spec_resp_agg = dataset.spec_resp_agg

    if arena_sizes is None:
        if spec_resp_agg is not None:
            arena_sizes = list(spec_resp_agg.keys())
        else:
            arena_sizes = list(np.unique(dataset["arena_size_cm"]))
    else:
        if not isinstance(arena_sizes, list):
            arena_sizes = [arena_sizes]

    scores_agg = {}
    for arena_size in arena_sizes:
        scores_agg[arena_size] = {}

        if spec_resp_agg is not None:
            for animal in spec_resp_agg[arena_size].keys():
                curr_resp = spec_resp_agg[arena_size][animal]["resp"]
                curr_cell_ids = spec_resp_agg[arena_size][animal]["cell_ids"]
                assert(len(curr_resp.shape) == 3)
                assert(curr_resp.shape[-1] == len(curr_cell_ids))
                # putting units in the first dimension for this function
                curr_animal_scores = compute_border_score_solstad(np.transpose(curr_resp, (2, 0, 1)),
                                                                  **border_score_params)

                scores_agg[arena_size][animal] = package_scores(scores=curr_animal_scores,
                                                                cell_ids=curr_cell_ids)

        else:
            for animal in np.unique(curr_arena_dataset["animal_id"]):
                curr_animal_scores = []
                cell_ids = []
                curr_animal_arena_dataset = curr_arena_dataset[curr_arena_dataset["animal_id"] == animal]
                for cell_idx in range(curr_animal_arena_dataset.shape[0]):
                    cell_ids.append(curr_animal_arena_dataset[cell_idx]["cell_id"])
                    # note that even cells within an animal can have different x and y positions, so we bin per cell separately
                    ret_val = compute_binned_frs(cell_idx=cell_idx,
                                                   curr_arena_dataset=curr_animal_arena_dataset,
                                                   arena_x_bins=arena_x_bins,
                                                   arena_y_bins=arena_y_bins,
                                                   **kwargs)
                    curr_border_score = compute_border_score_solstad(ret_val, **border_score_params)
                    curr_animal_scores.append(np.squeeze(curr_border_score))

                scores_agg[arena_size][animal] = package_scores(scores=np.stack(curr_animal_scores, axis=-1),
                                                                cell_ids=np.array(cell_ids))

    return scores_agg

def borderscore_permtest(dataset,
                        arena_sizes=None,
                        nbins_max=20,
                        border_score_params={},
                        n_perm=500, sig_alpha=0.02,
                        n_jobs=8,
                        **kwargs):

    from joblib import delayed, Parallel

    """
    Performs a border score permutation test across cells in a given animal, per arena.
    Note, this can be very slow so recommended to use the parallelized sherlock version that runs one job per cell.
    Only use this function if your neural data has a smaller number of cells or sherlock is down!
    Otherwise, the aggregate_borderscores function is recommended for efficiency (and take cells with > 0.5 border score, as Solstad et al. 2007 does).

    Does a permutation test, based on:
    https://github.com/alexgonzl/TreeMazeAnalyses2/blob/e4f92406b5e3b5281c5d092515ee10093be216ea/Analyses/spatial_functions.py#L1575-L1611.
    Default statistical testing kwargs from:
    https://github.com/alexgonzl/TreeMazeAnalyses2/blob/e4f92406b5e3b5281c5d092515ee10093be216ea/Analyses/open_field_functions.py#L1349-L1350
    """

    def p_worker(cell_idx,
                 curr_arena_dataset,
                 arena_x_bins,
                 arena_y_bins):
        """ helper function for parallelization. Computes a single shuffled border score per unit."""

        # get shifted rate map
        p_fr_map = compute_binned_frs(cell_idx=cell_idx,
                                       curr_arena_dataset=curr_animal_arena_dataset,
                                       arena_x_bins=arena_x_bins,
                                       arena_y_bins=arena_y_bins,
                                       shift=True,
                                       **kwargs)
        # get single border score
        p_bs = compute_border_score_solstad(p_fr_map, **border_score_params)
        return p_bs

    if arena_sizes is None:
        arena_sizes = list(np.unique(dataset["arena_size_cm"]))
    else:
        if not isinstance(arena_sizes, list):
            arena_sizes = [arena_sizes]

    scores_agg = {}
    for arena_size in arena_sizes:
        scores_agg[arena_size] = {}
        curr_arena_dataset = dataset[dataset["arena_size_cm"] == arena_size]
        arena_x_bins, arena_y_bins = get_xy_bins(curr_arena_dataset, nbins_max=nbins_max)

        for animal in np.unique(curr_arena_dataset["animal_id"]):
            curr_animal_scores = []
            curr_animal_sigs = []
            cell_ids = []
            curr_animal_arena_dataset = curr_arena_dataset[curr_arena_dataset["animal_id"] == animal]
            for cell_idx in range(curr_animal_arena_dataset.shape[0]):
                cell_ids.append(curr_animal_arena_dataset[cell_idx]["cell_id"])
                # note that even cells within an animal can have different x and y positions, so we bin per cell separately
                ret_val = compute_binned_frs(cell_idx=cell_idx,
                                               curr_arena_dataset=curr_animal_arena_dataset,
                                               arena_x_bins=arena_x_bins,
                                               arena_y_bins=arena_y_bins,
                                               **kwargs)
                true_bs = compute_border_score_solstad(ret_val, **border_score_params)
                curr_animal_scores.append(true_bs)

                if not np.isnan(true_bs):
                    # get border score shuffle dist
                    perm_bs = Parallel(n_jobs=n_jobs)(delayed(p_worker)(cell_idx=cell_idx,
                                                                        curr_arena_dataset=curr_animal_arena_dataset,
                                                                        arena_x_bins=arena_x_bins,
                                                                        arena_y_bins=arena_y_bins) for _ in range(n_perm))
                    # find location of true gs
                    loc = np.array(perm_bs >= true_bs).mean()
                    # determine if outside distribution @ alpha level
                    sig = np.logical_or(loc <= sig_alpha / 2, loc >= 1 - sig_alpha / 2)
                else:
                    sig = False
                curr_animal_sigs.append(sig)

            scores_agg[arena_size][animal] = {"scores": package_scores(scores=np.stack(curr_animal_scores, axis=-1),
                                                            cell_ids=np.array(cell_ids)),
                                              "sigs": package_scores(scores=np.stack(curr_animal_sigs, axis=-1),
                                                            cell_ids=np.array(cell_ids))}

    return scores_agg

def aggregate_hdscores(dataset,
                    arena_sizes=None,
                    nbins_max=20,
                    min_speed=2,
                    max_speed=80,
                    n_perm=500,
                    sig_alpha=0.02,
                    n_jobs=8,
                    **kwargs):

    from joblib import delayed, Parallel

    """
    Aggregates head direction scores across cells in a given animal, per arena.
    This currently only works with Caitlin's data where we can access the animal's speed in a specific way.
    TODO: generalize to other datasets as need be.
    This function is adapted from:
    https://github.com/alexgonzl/TreeMazeAnalyses2/blob/e4f92406b5e3b5281c5d092515ee10093be216ea/Analyses/spatial_functions.py#L1136-L1151
    Taking default min speed and max speed from:
    https://github.com/alexgonzl/TreeMazeAnalyses2/blob/e4f92406b5e3b5281c5d092515ee10093be216ea/Analyses/open_field_functions.py#L1334-L1335 in cm/s
    Default statistical testing kwargs from:
    https://github.com/alexgonzl/TreeMazeAnalyses2/blob/e4f92406b5e3b5281c5d092515ee10093be216ea/Analyses/open_field_functions.py#L1349-L1350
    """

    def p_worker(cell_fr, cell_theta):
        """ helper function for parallelization. Computes a single shuffled hd score per unit."""

        # get permuted rate
        p_fr = np.random.permutation(cell_fr)
        # get single hd score
        p_hds, _, _, _, = resultant_vector_length(alpha=cell_theta, w=p_fr)
        return p_hds

    if arena_sizes is None:
        arena_sizes = list(np.unique(dataset["arena_size_cm"]))
    else:
        if not isinstance(arena_sizes, list):
            arena_sizes = [arena_sizes]

    scores_agg = {}
    for arena_size in arena_sizes:
        scores_agg[arena_size] = {}
        curr_arena_dataset = dataset[dataset["arena_size_cm"] == arena_size]
        arena_x_bins, arena_y_bins = get_xy_bins(curr_arena_dataset, nbins_max=nbins_max)

        for animal in np.unique(curr_arena_dataset["animal_id"]):
            curr_animal_scores = []
            curr_animal_sigs = []
            cell_ids = []
            curr_animal_arena_dataset = curr_arena_dataset[curr_arena_dataset["animal_id"] == animal]
            for cell_idx in range(curr_animal_arena_dataset.shape[0]):
                cell_ids.append(curr_animal_arena_dataset[cell_idx]["cell_id"])
                # cm/s
                body_speed = curr_animal_arena_dataset[cell_idx]["body_speed"]
                # converting from degrees to radians
                theta = np.deg2rad(curr_animal_arena_dataset[cell_idx]["azimuthal_head_direction"])
                fr = compute_binned_frs(cell_idx=cell_idx,
                                        curr_arena_dataset=curr_animal_arena_dataset,
                                        arena_x_bins=arena_x_bins,
                                        arena_y_bins=arena_y_bins,
                                        return_unbinned=True)
                n_samps = len(theta)
                assert(fr.ndim == 1)
                assert(n_samps == len(fr))
                assert(n_samps == len(body_speed))

                if (min_speed is not None) and (max_speed is not None):
                    valid_samps = np.logical_and(body_speed >= min_speed, body_speed <= max_speed)
                    theta = theta[valid_samps]
                    fr = fr[valid_samps]

                true_hds, _, _, _, = resultant_vector_length(alpha=theta, w=fr)
                curr_animal_scores.append(true_hds)

                # get hd score shuffle dist
                perm_hds = Parallel(n_jobs=n_jobs)(delayed(p_worker)(cell_fr=fr, cell_theta=theta) for _ in range(n_perm))
                # find location of true gs
                loc = np.array(perm_hds >= true_hds).mean()
                # determine if outside distribution @ alpha level
                sig = np.logical_or(loc <= sig_alpha / 2, loc >= 1 - sig_alpha / 2)
                curr_animal_sigs.append(sig)

            scores_agg[arena_size][animal] = {"scores": package_scores(scores=np.stack(curr_animal_scores, axis=-1),
                                                            cell_ids=np.array(cell_ids)),
                                              "sigs": package_scores(scores=np.stack(curr_animal_sigs, axis=-1),
                                                            cell_ids=np.array(cell_ids))}

    return scores_agg

def unit_concat(d, arena_size, inner_key=None):
    """For a given arena size, concatenates scalar values (e.g. neural predictivity of a model)
    for units across animals."""
    if inner_key is None:
        # e.g. grid scores
        return xr.concat([d[arena_size][a] for a in d[arena_size].keys()], dim="units")
    else:
        # e.g. neural predictivity of a model, so inner_key=metric, like "corr"
        return xr.concat([d[arena_size][a][inner_key] for a in d[arena_size].keys() if inner_key in d[arena_size][a].keys()], dim="units")

def get_trial_bounds(curr_cell_position,
                     start_position_thresh=0,
                     end_position_thresh=395):
    '''For a given cell's position, returns the trial index start and end bounds for 1D data.'''
    curr_idx = 0
    curr_trial_start = 0
    trial_bounds = []
    while curr_idx < len(curr_cell_position):
        curr_cell_curr_position = curr_cell_position[curr_idx]
        if curr_idx > 0:
            curr_cell_prev_position = curr_cell_position[curr_idx-1]
        else:
            curr_cell_prev_position = curr_cell_curr_position

        # whenever we are not at the end of the array, the start of a trial
        # occurs when it is 0/negative after a previous nonzero position right before it
        # this deals with the fact that the position can have two consecutive 0s
        # or the end can be 400.11 followed by 400
        if ((curr_idx < len(curr_cell_position) - 1) and (curr_cell_curr_position <= start_position_thresh) and (curr_cell_prev_position > end_position_thresh)):
            curr_trial_end = curr_idx - 1
            trial_bounds.append((curr_trial_start, curr_trial_end))
            # update new start position
            curr_trial_start = curr_idx
        # the end of the position array is the final trial's end
        elif curr_idx == len(curr_cell_position) - 1:
            curr_trial_end = curr_idx
            trial_bounds.append((curr_trial_start, curr_trial_end))
        curr_idx += 1
    return trial_bounds

def get_position_bins_1d(dataset, nbins=80):
    # Compute min/max position across all animals the dataset
    # These are the bins to be used for all animals.
    body_pos_min = np.amin([np.nanmin(pos_val) for pos_val in dataset["body_position"]])
    body_pos_max = np.amax([np.nanmax(pos_val) for pos_val in dataset["body_position"]])

    # ~5 cm bins, strategy adopted from Low et al. 2020 of using 80 bins, so 20 bins per 100 cm
    # which is consistent with what we did in the 2d foraging data
    pos_bins = np.linspace(body_pos_min, body_pos_max, nbins+1, endpoint=True)
    return pos_bins

def check_consistent_1d(curr_rec_dataset):
    # checks that body position is the same within data
    # so that we can bin across cells within a single session
    # and only need to compute trial bounds from one cell
    ex_body_pos = curr_rec_dataset["body_position"][0]
    ex_time = curr_rec_dataset["time"][0]
    for cell_idx, body_pos in enumerate(curr_rec_dataset["body_position"]):
        assert(np.array_equal(body_pos, ex_body_pos) is True)
        assert(np.array_equal(ex_time, curr_rec_dataset["time"][cell_idx]) is True)

def bin_1d_frs_trials(curr_rec_dataset, pos_bins,
                      curr_rec_trial_bounds=None,
                      dt=0.02, smooth_std=1,
                      clip=True, normalize=True,
                      rtol=1e-10, atol=1e-13):

    check_consistent_1d(curr_rec_dataset)
    t = curr_rec_dataset["time"][0]
    body_pos = curr_rec_dataset["body_position"][0]
    if curr_rec_trial_bounds is None:
        curr_rec_trial_bounds = get_trial_bounds(curr_rec_dataset["body_position"][0])

    frs_trials = []
    for (trial_start_idx, trial_end_idx) in curr_rec_trial_bounds:
        curr_t = t[trial_start_idx:trial_end_idx+1]
        curr_body_pos = body_pos[trial_start_idx:trial_end_idx+1]
        # cells go in the inner loop since trials are of different lengths
        # and we want to bin across cells
        sp_counts_cells = []
        for cell_idx in range(len(curr_rec_dataset)):
            # spike times are cell-specific, of course
            spt = curr_rec_dataset["spike_times"][cell_idx]
            # since spike times are floats and our resolution is dt
            spt = np.round(spt/dt)*dt

            diff_t = np.diff(curr_t)
            # sometimes this can be 0.019999...
            assert(np.isclose(diff_t, dt*np.ones_like(diff_t)).all())

            # get the spike times within the current time interval
            curr_spt = spt[(np.amin(curr_t) <= spt) & (spt <= np.amax(curr_t))]
            # find the index in t of the time a spike occured
            # since we are manually rounding the spike times (rather than in earlier data it was rounded)
            # we use np.isclose instead of equality
            curr_t_indices = []
            for time_item_idx, time_item in enumerate(curr_t):
                for spt_item in curr_spt:
                    if np.isclose(time_item, spt_item,
                                  rtol=rtol, atol=atol):
                        curr_t_indices.append(time_item_idx)
                        break
            # find the counts of each spike time
            _, curr_c = np.unique(curr_spt, return_counts=True)
            curr_sp_counts = np.zeros(curr_t.shape)
            # associate the times with the counts
            curr_sp_counts[curr_t_indices] = curr_c
            assert(np.sum(curr_sp_counts) == curr_spt.shape[0])

            sp_counts_cells.append(curr_sp_counts)

        # num_cells x num_time
        sp_counts_cells = np.stack(sp_counts_cells, axis=0)

        # num_cells x positions
        binned_frs = binned_statistic(x=curr_body_pos,
                                      values=sp_counts_cells,
                                      statistic="mean",
                                      bins=pos_bins).statistic
        binned_avg_frs = binned_frs*(1.0/dt)
        # positions x num_cells
        binned_avg_frs = binned_avg_frs.T
        if smooth_std is not None:
            # gaussian filter applied only to the positions dimension
            binned_avg_frs = scipy.ndimage.gaussian_filter1d(input=np.nan_to_num(binned_avg_frs),
                                                             sigma=smooth_std, mode="constant", axis=0)
        frs_trials.append(binned_avg_frs)

    # trials x positions x num_cells
    # TODO: after testing, package into xarray
    frs_trials = np.stack(frs_trials, axis=0)
    return frs_trials

def trialbatch_1dvr_file_loader(session_name, num_files=None):
    if num_files is None:
        sess_numfiles = {}
        sess_numfiles[1] = 75
        sess_numfiles[2] = 105
        sess_numfiles[3] = 50
        sess_numfiles[4] = 75
        sess_numfiles[5] = 121
        sess_numfiles[6] = 75
        sess_numfiles[7] = 75
        sess_numfiles[8] = 50
        sess_numfiles[9] = 110
        sess_numfiles[10] = 110
        sess_numfiles[11] = 110

        num_files = sess_numfiles[session_name]

    full_data = []
    for i in range(num_files):
        curr_data = np.load(os.path.join(CAITLIN1D_VR_TRIALBATCH_DIR,
                                         f"caitlin_1d_vr_binned_session{session_name}_trialbatch{i+1}outof{num_files}.npz"))['arr_0'][()]
        full_data.append(curr_data)
    full_data = np.concatenate(full_data, axis=0)
    return full_data

def animal_session_1dvr(sessions=CAITLIN1D_VR_SESSIONS,
                        precomputed=True):
    if precomputed:
        print("Loading precomputed animal session data")
        animal_sess_mapping_ld = np.load(os.path.join(BASE_DIR_RESULTS, "caitlin1d_vr_animal_session_mapping.npz"),
                                      allow_pickle=True)['arr_0'][()]

        if sessions != CAITLIN1D_VR_SESSIONS:
            animal_sess_mapping = {}
            for spec, spec_sessions in animal_sess_mapping_ld.items():
                new_spec_sessions = []
                for s in spec_sessions:
                    if s in sessions:
                        # only include the relevant sessions
                        new_spec_sessions.append(s)
                animal_sess_mapping[spec] = new_spec_sessions
        else:
            animal_sess_mapping = animal_sess_mapping_ld
    else:
        dataset = pickle.load(open(CAITLIN1D_VR_PACKAGED, "rb"))
        animal_sess_mapping = {}
        for s in sessions:
            fn = f"cell_info_session{s}.mat"
            curr_rec_dataset = dataset[dataset["session_filename"] == fn]
            animals = np.unique(curr_rec_dataset["animal_id"])
            assert(len(animals) == 1)
            animal_id = animals[0]
            if animal_id not in animal_sess_mapping.keys():
                animal_sess_mapping[animal_id] = [s]
            else:
                animal_sess_mapping[animal_id].append(s)

    return animal_sess_mapping

def rename_cid(spec_resp_agg):
    # rename cell ids with animal
    spec_resp_agg_cellid = copy.deepcopy(spec_resp_agg)
    num_new_cell_ids = 0
    num_old_cell_ids = 0
    for arena_size in spec_resp_agg_cellid.keys():
        for animal in spec_resp_agg_cellid[arena_size].keys():
            new_cell_ids = []
            for curr_id in spec_resp_agg_cellid[arena_size][animal]['cell_ids']:
                new_cell_ids.append(f"{animal}_{curr_id}")
            spec_resp_agg_cellid[arena_size][animal]['cell_ids'] = np.array(new_cell_ids)
            num_new_cell_ids += len(np.unique(spec_resp_agg_cellid[arena_size][animal]['cell_ids']))
            num_old_cell_ids += len(np.unique(spec_resp_agg[arena_size][animal]['cell_ids']))
    assert(num_new_cell_ids == num_old_cell_ids)
    return spec_resp_agg_cellid
