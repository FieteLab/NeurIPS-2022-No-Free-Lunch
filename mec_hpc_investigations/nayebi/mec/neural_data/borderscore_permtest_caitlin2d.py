import numpy as np
import os
from mec_hpc_investigations.core.default_dirs import CAITLIN_BSCORES
from mec_hpc_investigations.core.utils import get_params_from_workernum, dict_to_str
from mec_hpc_investigations.neural_data.datasets import CaitlinDatasetWithoutInertial
from mec_hpc_investigations.neural_data.utils import get_xy_bins, compute_binned_frs
from mec_hpc_investigations.neural_data.border_score_utils import compute_border_score_solstad
from joblib import delayed, Parallel

def build_param_lookup(arena_sizes=[100], nbins_max=20,
                       border_score_params={},
                       n_perm=500, sig_alpha=0.02,
                       n_jobs=8,
                       smooth_std=1):

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
                                          "border_score_params": border_score_params,
                                          "n_perm": n_perm,
                                          "sig_alpha": sig_alpha,
                                          "n_jobs": n_jobs,
                                          "smooth_std": smooth_std
                                          }
                key += 1

    return param_lookup


def border_score_perm_test(cell_idx,
                           curr_animal_arena_dataset,
                           arena_x_bins,
                           arena_y_bins,
                           arena_size,
                           animal,
                           border_score_params,
                           n_perm,
                           sig_alpha,
                           n_jobs,
                           smooth_std=1):

    def p_worker():
        """ helper function for parallelization. Computes a single shuffled border score per unit."""

        # get shifted rate map
        p_fr_map = compute_binned_frs(cell_idx=cell_idx,
                                       curr_arena_dataset=curr_animal_arena_dataset,
                                       arena_x_bins=arena_x_bins,
                                       arena_y_bins=arena_y_bins,
                                       shift=True,
                                       smooth_std=smooth_std)
        # get single border score
        p_bs = compute_border_score_solstad(p_fr_map, **border_score_params)
        return p_bs

    cell_id = curr_animal_arena_dataset[cell_idx]["cell_id"]
    ret_val = compute_binned_frs(cell_idx=cell_idx,
                                 curr_arena_dataset=curr_animal_arena_dataset,
                                 arena_x_bins=arena_x_bins,
                                 arena_y_bins=arena_y_bins,
                                 smooth_std=smooth_std)
    true_bs = compute_border_score_solstad(ret_val, **border_score_params)
    if not np.isnan(true_bs):
        # get border score shuffle dist
        perm_bs = Parallel(n_jobs=n_jobs)(delayed(p_worker)() for _ in range(n_perm))
        # find location of true gs
        loc = np.array(perm_bs >= true_bs).mean()
        # determine if outside distribution @ alpha level
        sig = np.logical_or(loc <= sig_alpha / 2, loc >= 1 - sig_alpha / 2)
    else:
        sig = False

    results_dict = {"score": true_bs, "sig": sig, "cell_id": cell_id}
    fname = f"borderscores_nperm{n_perm}sigalpha{sig_alpha}_smoothstd{smooth_std}_{dict_to_str(border_score_params)}_caitlin2darena{arena_size}_{animal}_cell{cell_idx}.npz"
    filename = os.path.join(CAITLIN_BSCORES, fname)
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
    border_score_perm_test(**curr_params)
