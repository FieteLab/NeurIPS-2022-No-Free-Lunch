import os, copy
import numpy as np
import itertools
from mec_hpc_investigations.models.utils import configure_options
from mec_hpc_investigations.core.default_dirs import BASE_DIR_RESULTS, BASE_DIR_MODELS
from mec_hpc_investigations.core.utils import get_params_from_workernum, iterate_dicts, dict_to_str
from mec_hpc_investigations.neural_fits.comparisons import get_fits
from mec_hpc_investigations.neural_data.utils import unit_concat
from mec_hpc_investigations.neural_data.datasets import CaitlinDatasetWithoutInertial
from mec_hpc_investigations.neural_data.utils import aggregate_responses

RNN_archs = ["rnn", "lstm", "UGRNN", "GRU", "VanillaRNN"]
OTHERS = ["place_cells", "velocity_input", "nmf"]
ALL_TYPES = RNN_archs + OTHERS

def build_param_lookup(map_type,
                       map_kwargs={},
                       train_frac=0.2,
                       eval_arena_size=100,
                       ):
    common_kwargs = {"train_frac": train_frac,
                     "eval_arena_size": eval_arena_size,
                     "map_type": map_type,
                     "map_kwargs": map_kwargs}
    param_lookup = {}
    key = 0
    for model_type in ALL_TYPES:
        if model_type in ["place_cells", "velocity_input"]:
            model_name = model_type
            curr_kwargs = copy.deepcopy(common_kwargs)
            curr_kwargs["model_name"] = model_name
            curr_kwargs["model_type"] = model_type
            param_lookup[str(key)] = curr_kwargs
            key += 1
        elif model_type == "nmf":
            for nmf_components in [9, 100, 256, 512]:
                for mode in [eval_arena_size, "original"]:
                    model_name = f"{model_type}_{nmf_components}components_{mode}"
                    curr_kwargs = copy.deepcopy(common_kwargs)
                    curr_kwargs["model_name"] = model_name
                    curr_kwargs["model_type"] = model_type
                    param_lookup[str(key)] = curr_kwargs
                    key += 1
        else:
            for mode in ["random", eval_arena_size, "original", "pos"]:
                for activation in ["linear", "tanh", "relu", "sigmoid"]:
                    model_name = f"{model_type}_{activation}_{mode}"
                    curr_kwargs = copy.deepcopy(common_kwargs)
                    curr_kwargs["model_name"] = model_name
                    curr_kwargs["model_type"] = model_type
                    param_lookup[str(key)] = curr_kwargs
                    key += 1
    return param_lookup

def compute_model_consistencies(train_frac,
                                map_type,
                                map_kwargs,
                                model_name,
                                model_type,
                                eval_arena_size=100,
                                num_train_test_splits=10):

    dataset_obj = CaitlinDatasetWithoutInertial()
    dataset_obj.package_data()
    dataset = dataset_obj.packaged_data

    spec_resp_agg = aggregate_responses(dataset=dataset,
                                        smooth_std=1)

    if model_type in RNN_archs:
        fn = os.path.join(BASE_DIR_MODELS, f"{model_name}_alllayeractivations_caitlin2darena{eval_arena_size}.npz")
        model_resp_dict = np.load(fn, allow_pickle=True)["arr_0"][()]
        results_dict = {}
        # we do fits per RNN layer
        for curr_layer, model_resp in model_resp_dict.items():
            results_dict[curr_layer] = get_fits(dataset,
                                                 model_resp=model_resp,
                                                 arena_sizes=[eval_arena_size],
                                                 spec_resp_agg=spec_resp_agg,
                                                 train_frac=train_frac,
                                                 map_type=map_type,
                                                 map_kwargs=map_kwargs)
    else:
        fn = os.path.join(BASE_DIR_MODELS, f"{model_name}_activations_caitlin2darena{eval_arena_size}.npz")
        model_resp = np.load(fn, allow_pickle=True)["arr_0"][()]
        results_dict = get_fits(dataset,
                                 model_resp=model_resp,
                                 arena_sizes=[eval_arena_size],
                                 spec_resp_agg=spec_resp_agg,
                                 train_frac=train_frac,
                                 map_type=map_type,
                                 map_kwargs=map_kwargs)

    suffix = f"results_caitlin2darena{eval_arena_size}_trainfrac{train_frac}_maptype{map_type}_{dict_to_str(map_kwargs)}_numtrsp{num_train_test_splits}"
    fname = f"{model_name}_{suffix}.npz"
    filename = os.path.join(BASE_DIR_RESULTS, fname)
    np.savez(filename, results_dict)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_type", type=str)
    parser.add_argument("--train_frac", type=float, default=0.2)
    parser.add_argument("--score_method", type=str, default=None)
    args = parser.parse_args()

    if (args.score_method is None) or (args.score_method == "angular"):
        map_kwargs = {}
    else:
        map_kwargs = {"score_method": args.score_method}

    print('Looking up params')
    param_lookup = build_param_lookup(map_type=args.map_type,
                                      map_kwargs=map_kwargs,
                                      train_frac=args.train_frac)
    print('NUM JOBS', len(list(param_lookup.keys())))
    curr_params = get_params_from_workernum(worker_num=os.environ.get('SLURM_ARRAY_TASK_ID'), param_lookup=param_lookup)
    print('Curr params', curr_params)
    compute_model_consistencies(**curr_params)
