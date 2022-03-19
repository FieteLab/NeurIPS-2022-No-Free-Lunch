import numpy as np
import os
from mec_hpc_investigations.core.default_dirs import CAITLIN_HDSCORES
from mec_hpc_investigations.core.utils import get_params_from_workernum, dict_to_str
from mec_hpc_investigations.neural_data.datasets import CaitlinDatasetWithoutInertial
from mec_hpc_investigations.neural_data.utils import get_xy_bins, compute_binned_frs
from mec_hpc_investigations.neural_data.head_direction_score_utils import resultant_vector_length
from joblib import delayed, Parallel

def build_param_lookup(arena_sizes=[100], nbins_max=20,
                       n_perm=500, sig_alpha=0.02,
                       min_speed=2, max_speed=80,
                       n_jobs=8):

    dataset_obj = CaitlinDatasetWithoutInertial()
    dataset_obj.package_data()
    dataset = dataset_obj.packaged_data

    if arena_sizes is None:
        arena_sizes = list(np.unique(dataset["arena_size_cm"]))
    else:
        if not isinstance(arena_sizes, list):
            arena_sizes = [arena_sizes]

    param_lookup = {}
    key = 0
    for arena_size in arena_sizes:
        curr_arena_dataset = dataset[dataset["arena_size_cm"] == arena_size]
        arena_x_bins, arena_y_bins = get_xy_bins(curr_arena_dataset, nbins_max=nbins_max)

        for animal in np.unique(curr_arena_dataset["animal_id"]):
            curr_animal_arena_dataset = curr_arena_dataset[curr_arena_dataset["animal_id"] == animal]

            for cell_idx in range(curr_animal_arena_dataset.shape[0]):
                param_lookup[str(key)] = {"cell_idx": cell_idx,
                                          "curr_animal_arena_dataset": curr_animal_arena_dataset,
                                          "arena_x_bins": arena_x_bins,
                                          "arena_y_bins": arena_y_bins,
                                          "arena_size": arena_size,
                                          "animal": animal,
                                          "min_speed": min_speed,
                                          "max_speed": max_speed,
                                          "n_perm": n_perm,
                                          "sig_alpha": sig_alpha,
                                          "n_jobs": n_jobs,
                                          }
                key += 1

    return param_lookup


def hd_score_perm_test(cell_idx,
                           curr_animal_arena_dataset,
                           arena_x_bins,
                           arena_y_bins,
                           arena_size,
                           animal,
                           n_perm,
                           sig_alpha,
                           n_jobs,
                           min_speed=2,
                           max_speed=80):
    """
    This function is adapted from:
    https://github.com/alexgonzl/TreeMazeAnalyses2/blob/e4f92406b5e3b5281c5d092515ee10093be216ea/Analyses/spatial_functions.py#L1136-L1151
    Taking default min speed and max speed from:
    https://github.com/alexgonzl/TreeMazeAnalyses2/blob/e4f92406b5e3b5281c5d092515ee10093be216ea/Analyses/open_field_functions.py#L1334-L1335 in cm/s"""

    cell_id = curr_animal_arena_dataset[cell_idx]["cell_id"]
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

    def p_worker():
        """ helper function for parallelization. Computes a single shuffled hd score per unit."""

        # get permuted rate
        p_fr = np.random.permutation(fr)
        # get single hd score
        p_hds, _, _, _, = resultant_vector_length(alpha=theta, w=p_fr)
        return p_hds

    # get hd score shuffle dist
    perm_hds = Parallel(n_jobs=n_jobs)(delayed(p_worker)() for _ in range(n_perm))
    # find location of true gs
    loc = np.array(perm_hds >= true_hds).mean()
    # determine if outside distribution @ alpha level
    sig = np.logical_or(loc <= sig_alpha / 2, loc >= 1 - sig_alpha / 2)

    results_dict = {"score": true_hds, "sig": sig, "cell_id": cell_id}
    fname = f"hdscores_nperm{n_perm}sigalpha{sig_alpha}_min{min_speed}max{max_speed}speedthresh_caitlin2darena{arena_size}_{animal}_cell{cell_idx}.npz"
    filename = os.path.join(CAITLIN_HDSCORES, fname)
    np.savez(filename, results_dict)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    param_lookup = build_param_lookup()
    print("NUM TOTAL JOBS", len(list(param_lookup.keys())))
    curr_params = get_params_from_workernum(worker_num=os.environ.get('SLURM_ARRAY_TASK_ID'),
                                            param_lookup=param_lookup)
    print("CURR PARAMS", curr_params)
    hd_score_perm_test(**curr_params)
