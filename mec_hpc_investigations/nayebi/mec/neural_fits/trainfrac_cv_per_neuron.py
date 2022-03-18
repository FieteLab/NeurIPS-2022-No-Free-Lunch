import os
import numpy as np

from mec_hpc_investigations.core.default_dirs import CAITLIN2D_INTERANIMAL_TRAINFRAC_RESULTS
from mec_hpc_investigations.core.constants import RIDGE_CV_ALPHA_LONG
from mec_hpc_investigations.core.utils import get_params_from_workernum, iterate_dicts
from mec_hpc_investigations.neural_fits.utils import generate_train_test_splits, nan_filter, package_scores
from mec_hpc_investigations.neural_mappers.cross_validator import CrossValidator
from mec_hpc_investigations.neural_data.datasets import CaitlinDatasetWithoutInertial
from mec_hpc_investigations.neural_data.utils import aggregate_responses

def build_param_lookup(target_animal,
                       trainfrac_range=list(np.linspace(0.01, 0.95, 376, endpoint=True)[1:]), # 0.01 is too small for 3 fold CV to work, so excluding it
                       arena_size=100,
                       smooth_std=1
                       ):

    dataset_obj = CaitlinDatasetWithoutInertial()
    dataset_obj.package_data()
    dataset = dataset_obj.packaged_data
    spec_resp_agg = aggregate_responses(dataset=dataset,
                                        smooth_std=smooth_std)
    arena_animals = list(spec_resp_agg[arena_size].keys())
    assert(target_animal in arena_animals)

    curr_source_animals = list(set(arena_animals) - set([target_animal]))
    assert(target_animal not in curr_source_animals)
    assert(len(curr_source_animals) == len(arena_animals) - 1)

    source_animal_resp = np.concatenate([spec_resp_agg[arena_size][source_animal]["resp"] for source_animal in curr_source_animals], axis=-1)
    target_animal_resp = spec_resp_agg[arena_size][target_animal]["resp"]
    num_target_neurons = target_animal_resp.shape[-1]
    # build param lookup
    param_lookup = {}
    key = 0
    for curr_trainfrac in trainfrac_range:
        param_lookup[str(key)] = {
                                  "target_resp": target_animal_resp.reshape(-1, target_animal_resp.shape[-1]),
                                  "source_resp": source_animal_resp.reshape(-1, source_animal_resp.shape[-1]),
                                  "target_animal": target_animal,
                                  "train_frac": curr_trainfrac,
                                  "target_cell_ids": spec_resp_agg[arena_size][target_animal]["cell_ids"]
                                  }
        key += 1
    return param_lookup

def optimize_per_neuron(target_resp,
                        source_resp,
                        target_animal,
                        target_cell_ids,
                        num_train_test_splits=10,
                        num_cv_splits=3,
                        neural_map_str="percentile",
                        train_frac=0.5,
                        alpha_range=RIDGE_CV_ALPHA_LONG):

    assert neural_map_str == "percentile"
    results = {}
    s = iterate_dicts({"percentile": [0],
                        "identity":[False],
                        "regression_type": ["Ridge"],
                        "regression_kwargs": [{"alpha": alpha} for alpha in alpha_range]})
    cv_params = construct_cv_params(neural_map_str=neural_map_str,
                                    params_iterator=s)

    results[f"trainfrac_{train_frac}"] = optimize_per_neuron_subroutine(target_resp=target_resp,
                                                                       source_resp=source_resp,
                                                                       cell_ids=target_cell_ids,
                                                                       cv_params=cv_params,
                                                                       target_animal=target_animal,
                                                                       train_frac=train_frac,
                                                                       num_train_test_splits=num_train_test_splits,
                                                                       num_cv_splits=num_cv_splits,
                                                                       neural_map_str=neural_map_str)

    return results

def optimize_per_neuron_subroutine(target_resp,
                                   source_resp,
                                   target_animal,
                                    cell_ids,
                                    cv_params,
                                    train_frac=0.5,
                                    num_train_test_splits=10,
                                    num_cv_splits=3,
                                    neural_map_str="percentile"):

    num_target_neurons = target_resp.shape[-1]
    assert(num_target_neurons == len(cell_ids))
    # run on sherlock per target neuron
    results = {}
    for curr_target_neuron in list(range(num_target_neurons)):
        results[curr_target_neuron] = {"train_scores": [],
                                       "test_scores": [],
                                       "best_params": [],
                                       "val_scores": []}

        Y = np.expand_dims(target_resp[:, curr_target_neuron], axis=-1)
        X = source_resp
        X, Y = nan_filter(X, Y)
        # we generate train/test splits for each source, target pair
        # since the stimuli can be different for each pair (when we don't smooth firing rates)
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

            M = CrossValidator(neural_map_str=neural_map_str,
                          cv_params=cv_params,
                          n_cv_splits=num_cv_splits)

            M.fit(X_train, Y_train)
            Y_train_pred = M.predict(X_train)
            train_score = M.neural_mapper.score(Y_train_pred, Y_train)
            Y_test_pred = M.predict(X_test)
            test_score = M.neural_mapper.score(Y_test_pred, Y_test)

            results[curr_target_neuron]["train_scores"].append(package_scores(train_score, np.array([cell_ids[curr_target_neuron]])))
            results[curr_target_neuron]["test_scores"].append(package_scores(test_score, np.array([cell_ids[curr_target_neuron]])))
            results[curr_target_neuron]["best_params"].append(M.best_params)
            results[curr_target_neuron]["val_scores"].append(M.best_scores)

    return results

def construct_cv_params(neural_map_str,
                        params_iterator):
    return [{"map_type": neural_map_str, "map_kwargs": p} for p in params_iterator]

def construct_filename(target_animal,
                       dataset_name="caitlin2dwithoutinertial",
                       arena_size=100,
                       smooth_std=1,
                       train_frac=0.5,
                       num_train_test_splits=10,
                       num_cv_splits=3,
                       neural_map_str="percentile",
                  ):

    fname = "optimize_per_neuron"
    fname += f"_dataset{dataset_name}"
    fname += f"_arenasize{arena_size}"
    fname += f"_smoothstd_{smooth_std}"
    fname += f"_targetanimal{target_animal}"
    fname += f"_maptype{neural_map_str}"
    fname += f"_trainfrac{train_frac}"
    fname += f"_numtrsp{num_train_test_splits}_numcvsp{num_cv_splits}"
    fname += ".npz"
    return fname

def save_results(fit_results,
                 target_animal,
                 dataset_name="caitlin2dwithoutinertial",
                  arena_size=100,
                  smooth_std=1,
                    train_frac=0.5,
                    num_train_test_splits=10,
                    num_cv_splits=3,
                    neural_map_str="percentile",
                    **kwargs
                  ):

    print(f"Saving results to this directory {CAITLIN2D_INTERANIMAL_TRAINFRAC_RESULTS}")
    fname = construct_filename(target_animal=target_animal,
                             dataset_name=dataset_name,
                              arena_size=arena_size,
                              smooth_std=smooth_std,
                                train_frac=train_frac,
                                num_train_test_splits=num_train_test_splits,
                                num_cv_splits=num_cv_splits,
                                neural_map_str=neural_map_str,
                  )

    np.savez(os.path.join(CAITLIN2D_INTERANIMAL_TRAINFRAC_RESULTS, fname), fit_results)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_animal", type=str)
    args = parser.parse_args()

    print('Looking up params')
    param_lookup = build_param_lookup(target_animal=args.target_animal)
    print('NUM JOBS', len(list(param_lookup.keys())))
    curr_params = get_params_from_workernum(worker_num=os.environ.get('SLURM_ARRAY_TASK_ID'), param_lookup=param_lookup)
    print('Curr params', curr_params)
    fit_results = optimize_per_neuron(**curr_params)
    save_results(fit_results,
                 **curr_params)
