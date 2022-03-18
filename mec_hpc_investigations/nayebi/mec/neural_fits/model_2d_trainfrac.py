import numpy as np
import os
from mec_hpc_investigations.models.utils import configure_options
from mec_hpc_investigations.core.utils import dict_to_str, get_params_from_workernum, get_kw_from_map
from mec_hpc_investigations.core.default_dirs import BASE_DIR_RESULTS, BASE_DIR_MODELS
from mec_hpc_investigations.neural_fits.comparisons import get_fits
from mec_hpc_investigations.neural_data.utils import unit_concat
from mec_hpc_investigations.neural_data.datasets import CaitlinDatasetWithoutInertial
from mec_hpc_investigations.neural_data.utils import aggregate_responses

def build_param_lookup(train_fracs=[0.1, 0.25, 0.4, 0.5],
                       map_types=["1-1", "ridge", "pls100"],
                       eval_arena_size=100
                       ):

    param_lookup = {}
    key = 0
    for map_type in map_types:
        map_kwargs = get_kw_from_map(map_type=map_type)
        for train_frac in train_fracs:

            for model_type in ["rnn", "lstm", "UGRNN", "place_cells", "place_cells_mec", "velocity_input", "nmf"]:
                if model_type in ["place_cells", "place_cells_mec", "velocity_input"]:
                    model_name = model_type
                    param_lookup[str(key)] = {"train_frac": train_frac,
                                              "map_kwargs": map_kwargs,
                                              "model_name": model_name,
                                              "eval_arena_size": eval_arena_size
                                              }
                    key += 1
                elif model_type == "nmf":
                    for nmf_components in [9, 100, 256, 512]:
                        for mode in [eval_arena_size, "original"]:
                            model_name = f"{model_type}_{nmf_components}components_{mode}"
                            param_lookup[str(key)] = {"train_frac": train_frac,
                                                      "map_kwargs": map_kwargs,
                                                      "model_name": model_name,
                                                      "eval_arena_size": eval_arena_size
                                                      }
                            key += 1
                else:
                    for mode in ["random", eval_arena_size, "original", "pos"]:
                        for activation in ["linear", "tanh", "relu"]:
                            model_name = f"{model_type}_{activation}_{mode}"
                            param_lookup[str(key)] = {"train_frac": train_frac,
                                                      "map_kwargs": map_kwargs,
                                                      "model_name": model_name,
                                                      "eval_arena_size": eval_arena_size
                                                      }
                            key += 1
    return param_lookup


def compute_model_consistencies(train_frac,
                                map_kwargs,
                                model_name,
                                eval_arena_size=100):

    dataset_obj = CaitlinDatasetWithoutInertial()
    dataset_obj.package_data()
    dataset = dataset_obj.packaged_data

    spec_resp_agg = aggregate_responses(dataset=dataset,
                                    smooth_std=1)

    if model_name == "place_cells_mec":
        # we have to just run this here since it is done in the fitting procedure
        curr_cfg_kwargs = {"arena_size": eval_arena_size,
                           "dataset": dataset}
        curr_cfg = configure_options(**curr_cfg_kwargs)
        results_dict = get_fits(dataset,
                                 arena_sizes=[eval_arena_size],
                                 spec_resp_agg=spec_resp_agg,
                                 train_frac=train_frac,
                                 cfg=curr_cfg,
                                 interanimal_mode="holdout",
                                 fit_place_cell_resp=True,
                                 **map_kwargs)
    else:
        fn = os.path.join(BASE_DIR_MODELS, f"{model_name}_activations_caitlin2darena{eval_arena_size}.npz")
        model_resp = np.load(fn)["arr_0"][()]
        results_dict = get_fits(dataset,
                                 model_resp=model_resp,
                                 arena_sizes=[eval_arena_size],
                                 spec_resp_agg=spec_resp_agg,
                                 train_frac=train_frac,
                                 **map_kwargs)

    fname = f"{model_name}_results_caitlin2darena{eval_arena_size}_trainfrac{train_frac}_{dict_to_str(map_kwargs)}.npz"
    filename = os.path.join(BASE_DIR_RESULTS, fname)
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
    compute_model_consistencies(**curr_params)
