import os
import numpy as np

from mec_hpc_investigations.core.default_dirs import BASE_DIR_RESULTS
from mec_hpc_investigations.core.utils import get_params_from_workernum, iterate_dicts
from mec_hpc_investigations.neural_fits.utils import generate_train_test_splits, nan_filter, package_scores
from mec_hpc_investigations.neural_mappers.cross_validator import CrossValidator
from mec_hpc_investigations.neural_data.datasets import CaitlinDatasetWithoutInertial
from mec_hpc_investigations.neural_data.utils import aggregate_responses

def build_param_lookup(target_neurons, loop_percentiles=False):
    # build param lookup
    param_lookup = {}
    key = 0
    for n in target_neurons:
        param_lookup[str(key)] = {'target_neuron_idxs': [n], 'loop_percentiles': loop_percentiles}
        key += 1
    return param_lookup

def get_neural_data(dataset_name="caitlin2dwithoutinertial",
                 arena_size=100,
                smooth_std=1):
    if dataset_name == "caitlin2dwithoutinertial":
        dataset_obj = CaitlinDatasetWithoutInertial()
    else:
        raise ValueError
    dataset_obj.package_data()
    dataset = dataset_obj.packaged_data
    spec_resp_agg = aggregate_responses(dataset=dataset,
                                        smooth_std=smooth_std)

    cell_ids = []
    flat_resp = []
    for animal in spec_resp_agg[arena_size].keys():
        flat_animal_resp = spec_resp_agg[arena_size][animal]["resp"].reshape(-1, spec_resp_agg[arena_size][animal]["resp"].shape[-1])
        flat_resp.append(flat_animal_resp)
        cell_ids.extend(spec_resp_agg[arena_size][animal]["cell_ids"])
    flat_resp = np.concatenate(flat_resp, axis=-1)
    cell_ids = np.array(cell_ids)
    return flat_resp, cell_ids

def optimize_per_neuron_subroutine(flat_resp,
                        cell_ids,
                        cv_params,
                    target_neuron_idxs,
                    train_frac=0.5,
                    num_train_test_splits=10,
                    num_cv_splits=5,
                    neural_map_str="percentile"):

    if not isinstance(target_neuron_idxs, list):
        target_neuron_idxs = [target_neuron_idxs]

    target_neurons = list(range(flat_resp.shape[-1]))
    assert(len(target_neurons) == len(cell_ids))
    # run on sherlock per target neuron
    results = {}
    for curr_target_neuron in target_neuron_idxs:
        results[curr_target_neuron] = {"train_scores": [],
                                       "test_scores": [],
                                       "best_params": [],
                                      "num_source_units": []}

        Y = np.expand_dims(flat_resp[:, curr_target_neuron], axis=-1)
        source_neurons = list(set(target_neurons) - set([curr_target_neuron]))
        assert(len(source_neurons) == len(target_neurons) - 1)
        X = flat_resp[:, source_neurons]
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
            results[curr_target_neuron]["num_source_units"].append(M.neural_mapper.map._num_source_units)

    return results

def construct_cv_params(neural_map_str,
                        params_iterator):
    return [{"map_type": neural_map_str, "map_kwargs": p} for p in params_iterator]

def optimize_per_neuron(flat_resp,
                        cell_ids,
                    target_neuron_idxs,
                    train_frac=0.5,
                    num_train_test_splits=10,
                    num_cv_splits=5,
                    neural_map_str="percentile",
                    percentile_range = list(np.arange(0, 100+1, 5)),
                    alpha_range = [1e-3, 1e-2, 1e-1, 0, 1, 1e1, 1e2, 1e3],
                    loop_percentiles=False):

    if loop_percentiles:
        assert neural_map_str == "percentile"
        results = {}
        # add in 1 neuron at a time
        for p in list(np.linspace(start=0, stop=100, num=len(cell_ids)-1, endpoint=True)):
            if p > 100:
                raise ValueError
            elif p < 100:
                s = iterate_dicts({"percentile": [p],
                                    "identity":[False], "alpha": alpha_range})
                cv_params = construct_cv_params(neural_map_str=neural_map_str,
                                                params_iterator=s)

                results[f"percentile_{p}"] = optimize_per_neuron_subroutine(flat_resp=flat_resp,
                                               cell_ids=cell_ids,
                                               cv_params=cv_params,
                                               target_neuron_idxs=target_neuron_idxs,
                                               train_frac=train_frac,
                                               num_train_test_splits=num_train_test_splits,
                                               num_cv_splits=num_cv_splits,
                                               neural_map_str=neural_map_str)
            else:
                s1 = iterate_dicts({"percentile": [p],
                                    "identity":[False], "alpha": alpha_range})
                cv_params1 = construct_cv_params(neural_map_str=neural_map_str,
                                                params_iterator=s1)

                results[f"percentile_{p}"] = optimize_per_neuron_subroutine(flat_resp=flat_resp,
                                               cell_ids=cell_ids,
                                               cv_params=cv_params1,
                                               target_neuron_idxs=target_neuron_idxs,
                                               train_frac=train_frac,
                                               num_train_test_splits=num_train_test_splits,
                                               num_cv_splits=num_cv_splits,
                                               neural_map_str=neural_map_str)

                # adding in option of identity map for percentile of 100
                s2 = iterate_dicts({"percentile": [p], "identity":[True]})
                cv_params2 = construct_cv_params(neural_map_str=neural_map_str,
                                                params_iterator=s2)

                results[f"percentile_{p}_identity"] = optimize_per_neuron_subroutine(flat_resp=flat_resp,
                                               cell_ids=cell_ids,
                                               cv_params=cv_params2,
                                               target_neuron_idxs=target_neuron_idxs,
                                               train_frac=train_frac,
                                               num_train_test_splits=num_train_test_splits,
                                               num_cv_splits=num_cv_splits,
                                               neural_map_str=neural_map_str)
    else:
        if neural_map_str == "percentile":
            s1 = iterate_dicts({"percentile": percentile_range,
                                "identity":[False], "alpha": alpha_range})
            s2 = iterate_dicts({"percentile": [100], "identity":[True]})
            s = s1 + s2
        else:
            raise ValueError

        cv_params = construct_cv_params(neural_map_str=neural_map_str,
                                        params_iterator=s)

        results = optimize_per_neuron_subroutine(flat_resp=flat_resp,
                                       cell_ids=cell_ids,
                                       cv_params=cv_params,
                                       target_neuron_idxs=target_neuron_idxs,
                                       train_frac=train_frac,
                                       num_train_test_splits=num_train_test_splits,
                                       num_cv_splits=num_cv_splits,
                                       neural_map_str=neural_map_str)

    return results

def construct_filename(target_neuron_idxs,
                 dataset_name="caitlin2dwithoutinertial",
                  arena_size=100,
                  smooth_std=1,
                    train_frac=0.5,
                    num_train_test_splits=10,
                    num_cv_splits=5,
                    neural_map_str="percentile",
                    loop_percentiles=False
                  ):

    if not isinstance(target_neuron_idxs, list):
        target_neuron_idxs = [target_neuron_idxs]

    fname = "optimize_per_neuron"
    fname += f"_dataset{dataset_name}"
    fname += f"_arenasize{arena_size}"
    fname += f"_smoothstd_{smooth_std}"
    fname += f"_targetneurons{target_neuron_idxs}"
    fname += f"_maptype{neural_map_str}"
    fname += f"_trainfrac{train_frac}"
    fname += f"_numtrsp{num_train_test_splits}_numcvsp{num_cv_splits}"
    if loop_percentiles:
        fname += "_looppercentilesTrue"
    fname += ".npz"
    return fname

def save_results(fit_results,
                 target_neuron_idxs,
                 dataset_name="caitlin2dwithoutinertial",
                  arena_size=100,
                  smooth_std=1,
                    train_frac=0.5,
                    num_train_test_splits=10,
                    num_cv_splits=5,
                    neural_map_str="percentile",
                    loop_percentiles=False
                  ):

    print(f"Saving results to this directory {BASE_DIR_RESULTS}")
    fname = construct_filename(target_neuron_idxs=target_neuron_idxs,
                 dataset_name=dataset_name,
                  arena_size=arena_size,
                  smooth_std=smooth_std,
                    train_frac=train_frac,
                    num_train_test_splits=num_train_test_splits,
                    num_cv_splits=num_cv_splits,
                    neural_map_str=neural_map_str,
                    loop_percentiles=loop_percentiles
                  )

    np.savez(os.path.join(BASE_DIR_RESULTS, fname), fit_results)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop_percentiles", type=bool, default=False)
    args = parser.parse_args()

    print("Getting neural data")
    flat_resp, cell_ids = get_neural_data()
    print('Looking up params')
    param_lookup = build_param_lookup(target_neurons=list(range(flat_resp.shape[-1])),
                                      loop_percentiles=args.loop_percentiles)
    curr_params = get_params_from_workernum(worker_num=os.environ.get('SLURM_ARRAY_TASK_ID'), param_lookup=param_lookup)
    print('Curr params', curr_params)
    fit_results = optimize_per_neuron(flat_resp, cell_ids, **curr_params)
    save_results(fit_results,
                 **curr_params)
