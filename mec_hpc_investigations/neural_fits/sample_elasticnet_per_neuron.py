import os
import numpy as np
import itertools
from mec_hpc_investigations.core.default_dirs import CAITLIN2D_INTERANIMAL_SAMPLE_RESULTS, OFREWARD_COMBINED_INTERANIMAL_SAMPLE_RESULTS, CAITLINHPC_INTERANIMAL_SAMPLE_RESULTS
from mec_hpc_investigations.core.utils import get_params_from_workernum, iterate_dicts
from mec_hpc_investigations.neural_fits.utils import generate_train_test_splits, prep_data_2d, package_scores
from mec_hpc_investigations.neural_mappers.cross_validator import CrossValidator
from mec_hpc_investigations.neural_mappers.pipeline_neural_map import PipelineNeuralMap
from mec_hpc_investigations.neural_data.datasets import CaitlinDatasetWithoutInertial, RewardDataset, CaitlinHPCDataset
from mec_hpc_investigations.neural_data.utils import aggregate_responses, concat_resp_conds
from joblib import delayed, Parallel

TRAIN_FRAC_RNG = list(np.linspace(0.015, 0.5, num=25, endpoint=True))
# ALPHA_RNG = ([1.0] + list(np.geomspace(1e-9, 1e9, num=99, endpoint=True)))
ALPHA_RNG = list(np.geomspace(1e-9, 1e9, num=99, endpoint=True)) # commenting out above since this already includes 1.0
L1_RATIO_RNG = ([1e-16] + list(np.linspace(0.01, 1.0, num=99, endpoint=True)))

def build_param_lookup(dataset_name="caitlin2dwithoutinertial",
                       train_frac_range=None,
                       val_frac=None,
                       smooth_std=1,
                       num_cv_splits=3,
                       n_jobs=20,
                       ):

    if dataset_name == "caitlin2dwithoutinertial":
        dataset_obj = CaitlinDatasetWithoutInertial()
        dataset_obj.package_data()
        dataset = dataset_obj.packaged_data
        spec_resp_agg = aggregate_responses(dataset=dataset,
                                            smooth_std=smooth_std)
        arena_size = 100
    elif dataset_name.startswith("caitlinhpc"):
        dataset_obj = CaitlinHPCDataset()
        dataset_obj.package_data()
        spec_resp_agg = dataset_obj.spec_resp_agg
        if dataset_name == "caitlinhpc62":
            arena_size = 62
        elif dataset_name == "caitlinhpc50":
            arena_size = 50
        else:
            raise ValueError
    else:
        arena_size = 150
        if dataset_name == "ofreward_combined":
            of_dataset = RewardDataset(dataset="free_foraging")
            of_dataset.package_data()

            task_dataset = RewardDataset(dataset="task")
            task_dataset.package_data()

            spec_resp_agg = concat_resp_conds(of_dataset, task_dataset)
        elif dataset_name == "of_only":
            of_dataset = RewardDataset(dataset="free_foraging")
            of_dataset.package_data()
            spec_resp_agg = of_dataset.spec_resp_agg
        elif dataset_name == "reward_only":
            task_dataset = RewardDataset(dataset="task")
            task_dataset.package_data()
            spec_resp_agg = task_dataset.spec_resp_agg
        else:
            raise ValueError

    arena_animals = list(spec_resp_agg[arena_size].keys())

    # build param lookup
    param_lookup = {}
    key = 0
    for target_animal in arena_animals:
        curr_source_animals = list(set(arena_animals) - set([target_animal]))
        assert(target_animal not in curr_source_animals)
        assert(len(curr_source_animals) == len(arena_animals) - 1)

        source_animal_resp = np.concatenate([spec_resp_agg[arena_size][source_animal]["resp"] for source_animal in curr_source_animals], axis=-1)
        target_animal_resp = spec_resp_agg[arena_size][target_animal]["resp"]
        num_target_neurons = target_animal_resp.shape[-1]

        for n in list(range(num_target_neurons)):
            curr_default_cell_id = spec_resp_agg[arena_size][target_animal]["cell_ids"][n]
            param_lookup[str(key)] = {"dataset_name": dataset_name,
                                      "target_resp": np.expand_dims(target_animal_resp[:, n], axis=-1) if dataset_name == "ofreward_combined" else np.expand_dims(target_animal_resp[:, :, n], axis=-1),
                                      "source_resp": source_animal_resp,
                                      "target_cell_id": curr_default_cell_id if dataset_name == "caitlin2dwithoutinertial" else f"{target_animal}_{curr_default_cell_id}",
                                      "train_frac_range": train_frac_range,
                                      "val_frac": val_frac,
                                      "num_cv_splits": num_cv_splits,
                                      "n_jobs": n_jobs,
                                      }
            key += 1
    return param_lookup

def sample_per_neuron(target_resp,
                        source_resp,
                        target_cell_id,
                        dataset_name="caitlin2dwithoutinertial",
                        num_train_test_splits=10,
                        num_cv_splits=3,
                        train_frac_range=None,
                        val_frac=None,
                        alpha_range=ALPHA_RNG,
                        l1_ratio_range=L1_RATIO_RNG,
                        n_jobs=20):

    if train_frac_range is None:
        train_frac_range = TRAIN_FRAC_RNG

    results = Parallel(n_jobs=n_jobs)(delayed(_sampler)
                       (target_resp=target_resp,
                        source_resp=source_resp,
                        target_cell_id=target_cell_id,
                        num_train_test_splits=num_train_test_splits,
                        num_cv_splits=num_cv_splits,
                        train_frac=train_frac,
                        val_frac=val_frac,
                        alpha=alpha,
                        l1_ratio=l1_ratio)
                        for train_frac, alpha, l1_ratio in list(itertools.product(train_frac_range, alpha_range, l1_ratio_range)))

    return results

def _sampler(target_resp,
           source_resp,
           target_cell_id,
            train_frac,
            alpha,
            l1_ratio,
            num_train_test_splits=10,
            num_cv_splits=3,
            val_frac=None,
            neural_map_str="percentile"):

    results = {"train_scores": [],
               "test_scores": [],
               "best_params": [],
               "val_scores": [],
               "alpha_renorm": [],
               "alpha": alpha,
               "l1_ratio": l1_ratio,
               "train_frac": train_frac}

    if val_frac is not None:
        results["val_frac"] = val_frac

    Y = target_resp
    X = source_resp
    X, Y = prep_data_2d(X, Y)

    # we generate train/test splits for each source, target pair
    # since the stimuli can be different for each pair (when we don't smooth firing rates)
    train_test_sp = generate_train_test_splits(num_states=X.shape[0],
                                               train_frac=train_frac,
                                               val_frac=val_frac,
                                               num_splits=num_train_test_splits,
                                              )

    for curr_sp_idx, curr_sp in enumerate(train_test_sp):
        train_idx = curr_sp["train"]
        test_idx = curr_sp["test"]
        X_train = X[train_idx]
        Y_train = Y[train_idx]
        X_test = X[test_idx]
        Y_test = Y[test_idx]

        # divide alpha by number of training samples to match alpha of sklearn Ridge
        # see: https://stackoverflow.com/questions/47365978/scikit-learn-elastic-net-approaching-ridge
        curr_alpha = alpha/((float)(X_train.shape[0]))
        results["alpha_renorm"].append(curr_alpha)
        if val_frac is not None:
            val_idx = curr_sp["val"]
            X_val = X[val_idx]
            Y_val = Y[val_idx]
            try:
                curr_map_kwargs = {"regression_type": "ElasticNet",
                                   "regression_kwargs": {"alpha": curr_alpha, "l1_ratio": l1_ratio}}
                M = PipelineNeuralMap(map_type="sklinear",
                                      map_kwargs=curr_map_kwargs)
                M.fit(X_train, Y_train)
                Y_train_pred = M.predict(X_train)
                train_score = M.score(Y_train_pred, Y_train)
                Y_val_pred = M.predict(X_val)
                val_score = M.score(Y_val_pred, Y_val)
                Y_test_pred = M.predict(X_test)
                test_score = M.score(Y_test_pred, Y_test)

                results["train_scores"].append(package_scores(train_score, np.array([target_cell_id])))
                results["test_scores"].append(package_scores(test_score, np.array([target_cell_id])))
                results["val_scores"].append(package_scores(val_score, np.array([target_cell_id])))
                results["best_params"].append(curr_map_kwargs)
            except:
                # some choices of hyperparameters may not work so we set this to nan and break in this case
                results["train_scores"] = np.nan
                results["test_scores"] = np.nan
                results["val_scores"] = np.nan
                break
        else:
            # we do a single sampled parameter and run it through 3 fold cv to get val performance on it
            s = iterate_dicts({"percentile": [0],
                                "identity":[False],
                                "regression_type": ["ElasticNet"],
                                "regression_kwargs": [{"alpha": curr_alpha, "l1_ratio": l1_ratio}]})
            cv_params = construct_cv_params(neural_map_str=neural_map_str,
                                            params_iterator=s)

            try:
                M = CrossValidator(neural_map_str=neural_map_str,
                              cv_params=cv_params,
                              n_cv_splits=num_cv_splits)

                M.fit(X_train, Y_train)
                Y_train_pred = M.predict(X_train)
                train_score = M.neural_mapper.score(Y_train_pred, Y_train)
                Y_test_pred = M.predict(X_test)
                test_score = M.neural_mapper.score(Y_test_pred, Y_test)

                results["train_scores"].append(package_scores(train_score, np.array([target_cell_id])))
                results["test_scores"].append(package_scores(test_score, np.array([target_cell_id])))
                results["best_params"].append(M.best_params)
                results["val_scores"].append(M.best_scores)
            except:
                # some choices of hyperparameters may not work so we set this to nan and break in this case
                results["train_scores"] = np.nan
                results["test_scores"] = np.nan
                results["val_scores"] = np.nan
                break

    return results

def construct_cv_params(neural_map_str,
                        params_iterator):
    return [{"map_type": neural_map_str, "map_kwargs": p} for p in params_iterator]

def construct_filename(target_cell_id,
                       dataset_name="caitlin2dwithoutinertial",
                       arena_size=100,
                       smooth_std=1,
                       num_train_test_splits=10,
                       num_cv_splits=3,
                       neural_map_str="percentile",
                       train_frac_range=None,
                       val_frac=None,
                  ):

    fname = "sample_per_neuron"
    fname += f"_dataset{dataset_name}"
    fname += f"_arenasize{arena_size}"
    fname += f"_smoothstd_{smooth_std}"
    fname += f"_targetcellid{target_cell_id}"
    if val_frac is None:
        fname += f"_maptype{neural_map_str}"
    fname += f"_numtrsp{num_train_test_splits}"
    if val_frac is not None:
        fname += f"_val_frac{val_frac}"
    else:
        fname += f"_numcvsp{num_cv_splits}"
    if train_frac_range is not None: # for custom train fracs
        fname += f"_trainfracrng{train_frac_range}"
    fname += ".npz"
    return fname

def save_results(fit_results,
                 target_cell_id,
                 dataset_name="caitlin2dwithoutinertial",
                  smooth_std=1,
                    num_train_test_splits=10,
                    num_cv_splits=3,
                    neural_map_str="percentile",
                    train_frac_range=None,
                    val_frac=None,
                    **kwargs
                  ):

    if dataset_name == "caitlin2dwithoutinertial":
        save_dir = CAITLIN2D_INTERANIMAL_SAMPLE_RESULTS
        arena_size = 100
    elif dataset_name.startswith("caitlinhpc"):
        save_dir = CAITLINHPC_INTERANIMAL_SAMPLE_RESULTS
        if dataset_name == "caitlinhpc62":
            arena_size = 62
        elif dataset_name == "caitlinhpc50":
            arena_size = 50
        else:
            raise ValueError
    else:
        assert(dataset_name in ["ofreward_combined", "of_only", "reward_only"])
        save_dir = OFREWARD_COMBINED_INTERANIMAL_SAMPLE_RESULTS
        arena_size = 150
    print(f"Saving results to this directory {save_dir}")
    fname = construct_filename(target_cell_id=target_cell_id,
                             dataset_name=dataset_name,
                              arena_size=arena_size,
                              smooth_std=smooth_std,
                                num_train_test_splits=num_train_test_splits,
                                num_cv_splits=num_cv_splits,
                                neural_map_str=neural_map_str,
                                train_frac_range=train_frac_range,
                                val_frac=val_frac,
                  )

    np.savez(os.path.join(save_dir, fname), fit_results)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="caitlin2dwithoutinertial")
    parser.add_argument("--train_frac_range", type=str, default=None)
    parser.add_argument("--val_frac", type=float, default=None)
    parser.add_argument("--num_cv_splits", type=int, default=3)
    parser.add_argument("--n_jobs", type=int, default=20)
    args = parser.parse_args()

    if args.train_frac_range is not None:
        train_frac_range = [(float)(tr_e) for tr_e in args.train_frac_range.split(",")]
    else:
        train_frac_range = None
    print('Looking up params')
    param_lookup = build_param_lookup(dataset_name=args.dataset_name,
                                      train_frac_range=train_frac_range,
                                      val_frac=args.val_frac,
                                      num_cv_splits=args.num_cv_splits,
                                      n_jobs=args.n_jobs)
    print('NUM JOBS', len(list(param_lookup.keys())))
    curr_params = get_params_from_workernum(worker_num=os.environ.get('SLURM_ARRAY_TASK_ID'), param_lookup=param_lookup)
    print('Curr params', curr_params)
    fit_results = sample_per_neuron(**curr_params)
    save_results(fit_results,
                 **curr_params)
