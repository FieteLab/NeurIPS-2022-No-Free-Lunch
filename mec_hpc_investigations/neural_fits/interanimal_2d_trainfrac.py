import numpy as np
import os
from mec_hpc_investigations.core.utils import dict_to_str, get_params_from_workernum, get_kw_from_map
from mec_hpc_investigations.core.default_dirs import BASE_DIR_RESULTS
from mec_hpc_investigations.neural_fits.comparisons import get_fits
from mec_hpc_investigations.neural_data.utils import unit_concat
from mec_hpc_investigations.neural_data.datasets import CaitlinDatasetWithoutInertial
from mec_hpc_investigations.neural_data.utils import aggregate_responses

def build_param_lookup(train_fracs=None,
                       map_types="ridge"):

    if train_fracs is None:
        train_fracs = list(np.linspace(0.01, 0.5, 100, endpoint=True))
    else:
        train_fracs = train_fracs.split(",")

    param_lookup = {}
    key = 0
    for map_type in map_types.split(","):
       map_kwargs = get_kw_from_map(map_type=map_type)
       for train_frac in train_fracs:
           if isinstance(train_frac, str):
               train_frac = float(train_frac)
           param_lookup[str(key)] = {"train_frac": train_frac, "map_kwargs": map_kwargs}
           key += 1

    return param_lookup


def compute_interanimal_consistencies(train_frac,
                                      map_kwargs):

    dataset_obj = CaitlinDatasetWithoutInertial()
    dataset_obj.package_data()
    dataset = dataset_obj.packaged_data

    spec_resp_agg = aggregate_responses(dataset=dataset,
                                    smooth_std=1)

    results_dict = get_fits(dataset,
                          arena_sizes=[100],
                          spec_resp_agg=spec_resp_agg,
                          train_frac=train_frac,
                          interanimal_mode="holdout",
                          **map_kwargs)
    fname = f"results_interanimal_arena100_trainfrac{train_frac}_holdout_{dict_to_str(map_kwargs)}.npz"
    filename = os.path.join(BASE_DIR_RESULTS, fname)
    np.savez(filename, results_dict)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_types", type=str, default="ridge")
    parser.add_argument("--train_fracs", type=str, default=None)
    args = parser.parse_args()
    
    param_lookup = build_param_lookup(map_types=args.map_types, train_fracs=args.train_fracs)
    print("NUM TOTAL JOBS", len(list(param_lookup.keys())))
    curr_params = get_params_from_workernum(worker_num=os.environ.get('SLURM_ARRAY_TASK_ID'),
                                            param_lookup=param_lookup)
    print("CURR PARAMS", curr_params)
    compute_interanimal_consistencies(**curr_params)
