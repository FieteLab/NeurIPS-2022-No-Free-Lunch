import os, copy
import numpy as np
import itertools
from mec_hpc_investigations.core.default_dirs import BASE_DIR_MODELS, MODEL_CV_RESULTS_CAITLIN2D
from mec_hpc_investigations.core.utils import get_params_from_workernum, iterate_dicts
from mec_hpc_investigations.neural_fits.utils import generate_train_test_splits, package_scores, prep_data_2d
from mec_hpc_investigations.neural_mappers.cross_validator import CrossValidator
from mec_hpc_investigations.neural_data.datasets import CaitlinDatasetWithoutInertial
from mec_hpc_investigations.neural_data.utils import aggregate_responses

ALPHA_RNG = np.geomspace(1e-9, 1e9, num=99, endpoint=True) # includes 1.0
L1_RATIO_RNG = np.array([1e-16] + list(np.linspace(0.01, 1.0, num=99, endpoint=True)))

def build_param_lookup(cv_type="lasso",
                       train_frac=0.1,
                       eval_arena_size=100,
                       cv_n_jobs=20
                       ):
    common_kwargs = {"train_frac": train_frac,
                     "eval_arena_size": eval_arena_size,
                     "cv_type": cv_type,
                     "cv_n_jobs": cv_n_jobs}
    param_lookup = {}
    key = 0
    for model_type in ["rnn", "lstm", "UGRNN", "place_cells", "place_cells_mec", "velocity_input", "nmf", "interanimal"]:
        if model_type in ["place_cells", "place_cells_mec", "velocity_input", "interanimal"]:
            model_name = model_type
            curr_kwargs = copy.deepcopy(common_kwargs)
            curr_kwargs["model_name"] = model_name
            param_lookup[str(key)] = curr_kwargs
            key += 1
        elif model_type == "nmf":
            for nmf_components in [9, 100, 256, 512]:
                for mode in [eval_arena_size, "original"]:
                    model_name = f"{model_type}_{nmf_components}components_{mode}"
                    curr_kwargs = copy.deepcopy(common_kwargs)
                    curr_kwargs["model_name"] = model_name
                    param_lookup[str(key)] = curr_kwargs
                    key += 1
        else:
            for mode in ["random", eval_arena_size, "original", "pos"]:
                for activation in ["linear", "tanh", "relu"]:
                    model_name = f"{model_type}_{activation}_{mode}"
                    curr_kwargs = copy.deepcopy(common_kwargs)
                    curr_kwargs["model_name"] = model_name
                    param_lookup[str(key)] = curr_kwargs
                    key += 1
    return param_lookup

def compute_model_consistencies(train_frac,
                                cv_type,
                                model_name,
                                eval_arena_size=100,
                                num_train_test_splits=10,
                                num_cv_splits=3,
                                cv_n_jobs=20):

    dataset_obj = CaitlinDatasetWithoutInertial()
    dataset_obj.package_data()
    dataset = dataset_obj.packaged_data

    spec_resp_agg = aggregate_responses(dataset=dataset,
                                    smooth_std=1)

    arena_animals = list(spec_resp_agg[eval_arena_size].keys())

    first_X_in = None
    if model_name == "place_cells_mec":
        pc_fn = os.path.join(BASE_DIR_MODELS, f"place_cells_activations_caitlin2darena{eval_arena_size}.npz")
        first_X_in = np.load(pc_fn)["arr_0"][()]

    model_resp = None
    if model_name not in ["place_cells_mec", "interanimal"]:
        fn = os.path.join(BASE_DIR_MODELS, f"{model_name}_activations_caitlin2darena{eval_arena_size}.npz")
        model_resp = np.load(fn)["arr_0"][()]

    fit_results = {}
    fit_results[eval_arena_size] = {}
    for target_animal in arena_animals:
        target_animal_resp = spec_resp_agg[eval_arena_size][target_animal]["resp"]
        target_animal_cell_ids = spec_resp_agg[eval_arena_size][target_animal]["cell_ids"]

        # we check based on the model response since if it is None, X_in changes for each target animal
        # otherwise, if the criterion was based on X_in that would be overwritten after the first loop
        if model_resp is None:
            assert(model_name in ["place_cells_mec", "interanimal"])
            curr_source_animals = list(set(arena_animals) - set([target_animal]))
            assert(target_animal not in curr_source_animals)
            assert(len(curr_source_animals) == len(arena_animals) - 1)
            X_in = np.concatenate([spec_resp_agg[eval_arena_size][source_animal]["resp"] for source_animal in curr_source_animals], axis=-1)
        else:
            X_in = model_resp

        first_X = None
        if first_X_in is not None:
            # we find common visited positions between between first_X_in and X_in
            first_X, X = prep_data_2d(X=first_X_in, Y=X_in)
        else:
            X = copy.deepcopy(X_in)
        Y_in = copy.deepcopy(target_animal_resp)
        # we find the additionally common visited positions between X and each Y
        X, Y_in = prep_data_2d(X=X, Y=Y_in)

        for n in range(Y_in.shape[-1]):
            curr_target_neuron = target_animal_cell_ids[n]
            fit_results[eval_arena_size][curr_target_neuron] = {"train_scores": [],
                                                               "test_scores": [],
                                                               "best_params": [],
                                                               "val_scores": []}

            Y = np.expand_dims(Y_in[:, n], axis=-1)

            train_test_sp = generate_train_test_splits(num_states=X.shape[0],
                                                       train_frac=train_frac,
                                                       num_splits=num_train_test_splits,
                                                      )

            for curr_sp_idx, curr_sp in enumerate(train_test_sp):
                train_idx = curr_sp["train"]
                test_idx = curr_sp["test"]
                X_train = X[train_idx]
                Y_train = Y[train_idx]
                X_test = X[test_idx]
                Y_test = Y[test_idx]
                if first_X_in is not None:
                    first_X_train = first_X[train_idx]
                    pred_train_in = first_X_train
                    first_X_test = first_X[test_idx]
                    pred_test_in = first_X_test
                else:
                    first_X_train = None
                    pred_train_in = X_train
                    pred_test_in = X_test

                cv_params = construct_cv_params(cv_type=cv_type,
                                                num_examples=pred_train_in.shape[0])
                M = CrossValidator(neural_map_str="sklinear",
                                   cv_params=cv_params,
                                   n_cv_splits=num_cv_splits,
                                   n_jobs=cv_n_jobs)
                M.fit(X=X_train, Y=Y_train, first_X=first_X_train)
                Y_train_pred = M.predict(pred_train_in)
                train_score = M.neural_mapper.score(Y_pred=Y_train_pred, Y=Y_train)
                Y_test_pred = M.predict(pred_test_in)
                test_score = M.neural_mapper.score(Y_pred=Y_test_pred, Y=Y_test)

                fit_results[eval_arena_size][curr_target_neuron]["train_scores"].append(train_score)
                fit_results[eval_arena_size][curr_target_neuron]["test_scores"].append(test_score)
                fit_results[eval_arena_size][curr_target_neuron]["best_params"].append(M.best_params)
                fit_results[eval_arena_size][curr_target_neuron]["val_scores"].append(M.best_scores)

    fname = f"{model_name}_results_caitlin2darena{eval_arena_size}_trainfrac{train_frac}_cvtype{cv_type}_numtrsp{num_train_test_splits}_numcvsp{num_cv_splits}.npz"
    filename = os.path.join(MODEL_CV_RESULTS_CAITLIN2D, fname)
    np.savez(filename, fit_results)

def construct_cv_params(cv_type,
                        num_examples):

    regression_type = ["ElasticNet"]
    RENORM_ALPHA_RNG = ALPHA_RNG/((float)(num_examples))
    if cv_type == "ridge":
        params_iterator = iterate_dicts({"regression_type": regression_type,
                                         "regression_kwargs": [{"alpha": alpha, "l1_ratio": 1e-16} for alpha in list(RENORM_ALPHA_RNG)]})
    elif cv_type == "ridge_skdefault": # as a sanity chek
        params_iterator = iterate_dicts({"regression_type": regression_type,
                                         "regression_kwargs": [{"alpha": 1.0/((float)(num_examples)), "l1_ratio": 1e-16}]})
    elif cv_type == "lasso":
        params_iterator = iterate_dicts({"regression_type": regression_type,
                                         "regression_kwargs": [{"alpha": alpha, "l1_ratio": 1.0} for alpha in list(RENORM_ALPHA_RNG)]})
    elif cv_type == "elasticnet":
        params_iterator = iterate_dicts({"regression_type": regression_type,
                                         "regression_kwargs": [{"alpha": alpha, "l1_ratio": l1_ratio} for alpha, l1_ratio in list(itertools.product(RENORM_ALPHA_RNG, L1_RATIO_RNG))]})
    else:
        raise ValueError

    return [{"map_type": "sklinear", "map_kwargs": p} for p in params_iterator]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv_type", type=str)
    parser.add_argument("--cv_n_jobs", type=int, default=20)
    args = parser.parse_args()

    print('Looking up params')
    param_lookup = build_param_lookup(cv_type=args.cv_type, cv_n_jobs=args.cv_n_jobs)
    print('NUM JOBS', len(list(param_lookup.keys())))
    curr_params = get_params_from_workernum(worker_num=os.environ.get('SLURM_ARRAY_TASK_ID'), param_lookup=param_lookup)
    print('Curr params', curr_params)
    compute_model_consistencies(**curr_params)
