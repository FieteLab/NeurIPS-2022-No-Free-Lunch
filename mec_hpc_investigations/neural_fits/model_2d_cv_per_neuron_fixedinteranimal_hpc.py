import os, copy
import numpy as np
import itertools
from mec_hpc_investigations.models.utils import configure_options
from mec_hpc_investigations.core.default_dirs import BASE_DIR_MODELS, CAITLIN2D_HPC, HPC_INTERANIMAL_CC_MAP, HPC_INTERANIMAL_CC_MAP_MODEL_RESULTS, BASE_DIR_HPC_RESULTS
from mec_hpc_investigations.core.utils import get_params_from_workernum, iterate_dicts
from mec_hpc_investigations.neural_fits.comparisons import get_fits
from mec_hpc_investigations.neural_data.utils import unit_concat, rename_cid
from mec_hpc_investigations.neural_data.datasets import CaitlinHPCDataset

RNN_archs = ["rnn", "lstm", "UGRNN", "GRU", "VanillaRNN"]
OTHERS = ["place_cells", "place_cells_mec", "velocity_input", "nmf"]
ALL_TYPES = RNN_archs + OTHERS

def build_param_lookup(eval_arena_size,
                       cv_type="elasticnet_max",
                       train_frac=0.2,
                       val_frac=None,
                       num_cv_splits=2
                       ):
    common_kwargs = {"train_frac": train_frac,
                     "val_frac": val_frac,
                     "eval_arena_size": eval_arena_size,
                     "num_cv_splits": num_cv_splits,
                     "cv_type": cv_type}
    param_lookup = {}
    key = 0
    for model_type in ALL_TYPES:
        if model_type in ["place_cells", "place_cells_mec", "velocity_input"]:
            model_name = model_type
            curr_kwargs = copy.deepcopy(common_kwargs)
            curr_kwargs["model_name"] = model_name
            curr_kwargs["model_type"] = model_type
            param_lookup[str(key)] = curr_kwargs
            key += 1
        elif model_type == "nmf":
            for nmf_components in [9, 100, 256, 512]:
                for mode in ["original"]:
                    model_name = f"{model_type}_{nmf_components}components_{mode}"
                    curr_kwargs = copy.deepcopy(common_kwargs)
                    curr_kwargs["model_name"] = model_name
                    curr_kwargs["model_type"] = model_type
                    param_lookup[str(key)] = curr_kwargs
                    key += 1
        else:
            for mode in ["random", "original", "pos"]:
                for activation in ["linear", "tanh", "relu", "sigmoid"]:
                    model_name = f"{model_type}_{activation}_{mode}"
                    curr_kwargs = copy.deepcopy(common_kwargs)
                    curr_kwargs["model_name"] = model_name
                    curr_kwargs["model_type"] = model_type
                    param_lookup[str(key)] = curr_kwargs
                    key += 1
    return param_lookup

def compute_model_consistencies(train_frac,
                                cv_type,
                                model_name,
                                model_type,
                                eval_arena_size,
                                val_frac=None,
                                map_type="sklinear",
                                num_train_test_splits=10,
                                num_cv_splits=2):

    dataset = CaitlinHPCDataset()
    dataset.package_data()
    spec_resp_agg = rename_cid(dataset.spec_resp_agg)

    model_results_save_dir = HPC_INTERANIMAL_CC_MAP_MODEL_RESULTS

    # load interanimal consistency parameters to use for the models
    suffix = f"{cv_type}_arena{eval_arena_size}_trainfrac{train_frac}"
    if val_frac is not None:
        suffix += f"_valfrac{val_frac}"
    suffix += f"_numtrsp{num_train_test_splits}_numcvsp{num_cv_splits}"
    map_kwargs_per_cell = np.load(os.path.join(HPC_INTERANIMAL_CC_MAP, f"{suffix}.npz"),
                                  allow_pickle=True)["arr_0"][()]


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
                                 map_type=map_type,
                                 map_kwargs_per_cell=map_kwargs_per_cell)
    else:
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
                                                     map_kwargs_per_cell=map_kwargs_per_cell)
        else:
            fn = os.path.join(BASE_DIR_MODELS, f"{model_name}_activations_caitlin2darena{eval_arena_size}.npz")
            model_resp = np.load(fn, allow_pickle=True)["arr_0"][()]
            results_dict = get_fits(dataset,
                                     model_resp=model_resp,
                                     arena_sizes=[eval_arena_size],
                                     spec_resp_agg=spec_resp_agg,
                                     train_frac=train_frac,
                                     map_type=map_type,
                                     map_kwargs_per_cell=map_kwargs_per_cell)

    save_suffix = f"{suffix}_fixedinteranimal"
    fname = f"{model_name}_{save_suffix}.npz"
    filename = os.path.join(model_results_save_dir, fname)
    np.savez(filename, results_dict)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arena_size", type=int, choices=[50, 62])
    parser.add_argument("--cv_type", type=str, default="elasticnet_max")
    parser.add_argument("--train_frac", type=float, default=0.2)
    parser.add_argument("--val_frac", type=float, default=None)
    parser.add_argument("--num_cv_splits", type=int, default=2)
    args = parser.parse_args()

    print('Looking up params')
    param_lookup = build_param_lookup(cv_type=args.cv_type,
                                      eval_arena_size=args.arena_size,
                                      train_frac=args.train_frac,
                                      val_frac=args.val_frac,
                                      num_cv_splits=args.num_cv_splits)
    print('NUM JOBS', len(list(param_lookup.keys())))
    curr_params = get_params_from_workernum(worker_num=os.environ.get('SLURM_ARRAY_TASK_ID'), param_lookup=param_lookup)
    print('Curr params', curr_params)
    compute_model_consistencies(**curr_params)
