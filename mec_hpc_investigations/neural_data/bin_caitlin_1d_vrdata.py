import os
import numpy as np
import pickle
from mec_hpc_investigations.neural_data.utils import get_position_bins_1d, bin_1d_frs_trials, get_trial_bounds, check_consistent_1d
from mec_hpc_investigations.core.default_dirs import CAITLIN1D_VR_PACKAGED, CAITLIN1D_VR_TRIALBATCH_DIR
from mec_hpc_investigations.core.utils import get_params_from_workernum, iterate_dicts

def build_param_lookup(dataset,
                       sess_idxs=range(1, 12),
                       trial_batch_len=2):

    pos_bins = get_position_bins_1d(dataset)
    # build param lookup
    param_lookup = {}
    key = 0
    for s in sess_idxs:
        fn = f"cell_info_session{s}.mat"
        curr_rec_dataset = dataset[dataset["session_filename"] == fn]
        check_consistent_1d(curr_rec_dataset)
        curr_rec_trial_bounds = get_trial_bounds(curr_rec_dataset["body_position"][0])
        if trial_batch_len is not None:
           num_trials = (float)(len(curr_rec_trial_bounds))
           num_batches = (int)(np.floor(num_trials / trial_batch_len))
           num_total_batches = num_batches
           num_leftover_trials = num_trials % trial_batch_len
           if num_leftover_trials > 0:
               num_total_batches = num_batches + 1
           for i in range(num_batches):
               fname = f"caitlin_1d_vr_binned_session{s}_trialbatch{i+1}outof{num_total_batches}.npz"
               curr_trial_bounds = curr_rec_trial_bounds[i*trial_batch_len:(i+1)*trial_batch_len]
               param_lookup[str(key)] = {"curr_rec_dataset": curr_rec_dataset,
                                         "curr_trial_bounds": curr_trial_bounds,
                                         "pos_bins": pos_bins,
                                         "fname": fname
                                         }
               key += 1

           if num_leftover_trials > 0:
               fname = f"caitlin_1d_vr_binned_session{s}_trialbatch{num_total_batches}outof{num_total_batches}.npz"
               curr_trial_bounds = curr_rec_trial_bounds[num_batches*trial_batch_len:]
               param_lookup[str(key)] = {"curr_rec_dataset": curr_rec_dataset,
                                         "curr_trial_bounds": curr_trial_bounds,
                                         "pos_bins": pos_bins,
                                         "fname": fname
                                         }
               key += 1

        else:
            fname = f"caitlin_1d_vr_binned_session{s}_alltrials.npz"
            param_lookup[str(key)] = {"curr_rec_dataset": curr_rec_dataset,
                                      "curr_trial_bounds": curr_rec_trial_bounds,
                                      "pos_bins": pos_bins,
                                      "fname": fname
                                      }
            key += 1

    return param_lookup

def run_packaging(curr_rec_dataset,
                      curr_trial_bounds,
                      pos_bins,
                      fname):
    ret_dict = bin_1d_frs_trials(curr_rec_dataset=curr_rec_dataset,
                                 pos_bins=pos_bins,
                                 curr_rec_trial_bounds=curr_trial_bounds)
    np.savez(os.path.join(CAITLIN1D_VR_TRIALBATCH_DIR, fname), ret_dict)

if __name__ == '__main__':
    print("Loading neural data")
    dataset = pickle.load(open(CAITLIN1D_VR_PACKAGED, "rb"))
    print("Looking up params")
    param_lookup = build_param_lookup(dataset)
    print("NUM TOTAL JOBS", len(list(param_lookup.keys())))
    curr_params = get_params_from_workernum(worker_num=os.environ.get('SLURM_ARRAY_TASK_ID'), param_lookup=param_lookup)
    print("Curr params", curr_params)
    run_packaging(**curr_params)
