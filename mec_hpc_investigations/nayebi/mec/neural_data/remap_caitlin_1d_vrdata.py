import os
import numpy as np
from mec_hpc_investigations.neural_data.utils import trialbatch_1dvr_file_loader
from mec_hpc_investigations.neural_data.remap_utils import compute_num_maps
from mec_hpc_investigations.core.utils import get_params_from_workernum
from mec_hpc_investigations.core.default_dirs import BASE_DIR_RESULTS

def build_param_lookup(sess_idxs=range(1, 12),
                       ranks=range(1, 11)):
    # build param lookup
    param_lookup = {}
    key = 0
    for s in sess_idxs:
        for r in ranks:
            param_lookup[str(key)] = {"sess_name": s,
                                      "ranks": [r]}

            key += 1

    return param_lookup

def run_remap_analysis(sess_name,
                       ranks):
    sess_data = trialbatch_1dvr_file_loader(session_name=sess_name)
    metrics = compute_num_maps(sess_data,
                               clip=None,
                               normalize=False,
                               ranks=ranks,
                               max_iter=100)
    fname = f"caitlin1d_vr_data_remapanalysis_session{sess_name}_ranks{ranks}_maxiter100.npz"
    np.savez(os.path.join(BASE_DIR_RESULTS, fname), metrics)

if __name__ == '__main__':
    print("Looking up params")
    param_lookup = build_param_lookup()
    print("NUM TOTAL JOBS", len(list(param_lookup.keys())))
    curr_params = get_params_from_workernum(worker_num=os.environ.get('SLURM_ARRAY_TASK_ID'),
                                            param_lookup=param_lookup)
    print("Curr params", curr_params)
    run_remap_analysis(**curr_params)
