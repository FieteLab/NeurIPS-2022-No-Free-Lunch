import os, copy
import numpy as np
from netrep.metrics import LinearMetric
from mec_hpc_investigations.core.default_dirs import BASE_DIR_RESULTS
from mec_hpc_investigations.core.utils import get_params_from_workernum
from mec_hpc_investigations.neural_fits.utils import generate_train_test_splits, prep_data_2d
from mec_hpc_investigations.neural_data.datasets import CaitlinDatasetWithoutInertial, RewardDataset
from mec_hpc_investigations.neural_data.utils import aggregate_responses, concat_resp_conds
from joblib import delayed, Parallel
from sklearn.model_selection import KFold

ALPHA_SHAPE_RNG = list(np.linspace(0.0, 1.0, num=1000, endpoint=True))

def build_param_lookup(dataset_name="caitlin2dwithoutinertial",
                       train_frac=0.2,
                       val_frac=None,
                       smooth_std=1,
                       num_cv_splits=2,
                       ):

    if dataset_name == "caitlin2dwithoutinertial":
        dataset_obj = CaitlinDatasetWithoutInertial()
        dataset_obj.package_data()
        dataset = dataset_obj.packaged_data
        spec_resp_agg = aggregate_responses(dataset=dataset,
                                            smooth_std=smooth_std)
        arena_size = 100
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

    # build param lookup
    param_lookup = {}
    key = 0
    param_lookup[str(key)] = {"dataset_name": dataset_name,
                              "spec_resp_agg": spec_resp_agg[arena_size],
                              "train_frac": train_frac,
                              "val_frac": val_frac,
                              "num_cv_splits": num_cv_splits,
                              }
    return param_lookup

def sample_across_neurons(spec_resp_agg,
                        dataset_name="caitlin2dwithoutinertial",
                        num_train_test_splits=10,
                        num_cv_splits=2,
                        train_frac=0.2,
                        val_frac=None,
                        alpha_range=ALPHA_SHAPE_RNG,
                        n_jobs=20):

    results = Parallel(n_jobs=n_jobs)(delayed(_sampler)
                       (spec_resp_agg=spec_resp_agg,
                        num_train_test_splits=num_train_test_splits,
                        num_cv_splits=num_cv_splits,
                        train_frac=train_frac,
                        val_frac=val_frac,
                        alpha=alpha)
                        for alpha in alpha_range)

    return results

def _sampler(spec_resp_agg,
            train_frac,
            alpha,
            num_train_test_splits=10,
            num_cv_splits=2,
            val_frac=None):

    kf = KFold(n_splits=num_cv_splits)

    results = {"alpha": alpha,
               "train_frac": train_frac}

    if val_frac is not None:
        results["val_frac"] = val_frac

    arena_animals = list(spec_resp_agg.keys())

    for score_method in ["angular", "euclidean"]:
        train_dists = []
        test_dists = []
        val_dists = []
        for target_animal in arena_animals:
            curr_source_animals = list(set(arena_animals) - set([target_animal]))
            assert(target_animal not in curr_source_animals)
            assert(len(curr_source_animals) == len(arena_animals) - 1)

            source_animal_resp = np.concatenate([spec_resp_agg[source_animal]["resp"] for source_animal in curr_source_animals],
                                                axis=-1)
            target_animal_resp = spec_resp_agg[target_animal]["resp"]

            Y = target_animal_resp
            X = source_animal_resp
            X, Y = prep_data_2d(X, Y)

            # we generate train/test splits for each source, target pair
            # since the stimuli can be different for each pair (when we don't smooth firing rates)
            train_test_sp = generate_train_test_splits(num_states=X.shape[0],
                                                       train_frac=train_frac,
                                                       val_frac=val_frac,
                                                       num_splits=num_train_test_splits,
                                                      )

            train_dists_sp = []
            test_dists_sp = []
            val_dists_sp = []
            for curr_sp_idx, curr_sp in enumerate(train_test_sp):
                train_idx = curr_sp["train"]
                test_idx = curr_sp["test"]
                X_train = X[train_idx]
                Y_train = Y[train_idx]
                X_test = X[test_idx]
                Y_test = Y[test_idx]

                if val_frac is not None:
                    M = LinearMetric(alpha=alpha, score_method=score_method)
                    val_idx = curr_sp["val"]
                    X_val = X[val_idx]
                    Y_val = Y[val_idx]

                    M.fit(X_train, Y_train)
                    train_dist = M.score(X_train, Y_train)
                    val_dist = M.score(X_val, Y_val)
                    test_dist = M.score(X_test, Y_test)

                    train_dists_sp.append(train_dist)
                    val_dists_sp.append(val_dist)
                    test_dists_sp.append(test_dist)
                else:
                    # doing our own cv
                    val_dist_agg = np.nan
                    for cv_idx, (cv_train_idx, val_idx) in enumerate(kf.split(curr_sp['train'])):
                        M = LinearMetric(alpha=alpha, score_method=score_method)
                        X_train_cv = X[cv_train_idx]
                        Y_train_cv = Y[cv_train_idx]
                        X_val = X[val_idx]
                        Y_val = Y[val_idx]
                        M.fit(X_train_cv, Y_train_cv)
                        curr_val_dist = M.score(X_val, Y_val)
                        if cv_idx == 0:
                            val_dist_agg = copy.deepcopy(curr_val_dist)
                        else:
                            val_dist_agg = (val_dist_agg + curr_val_dist)

                    val_dist = (1.0/num_cv_splits)*val_dist_agg
                    M = LinearMetric(alpha=alpha, score_method=score_method)
                    M.fit(X_train, Y_train)
                    train_dist = M.score(X_train, Y_train)
                    test_dist = M.score(X_test, Y_test)

                    train_dists_sp.append(train_dist)
                    val_dists_sp.append(val_dist)
                    test_dists_sp.append(test_dist)

            # append across animals
            train_dists_sp = np.array(train_dists_sp)
            test_dists_sp = np.array(test_dists_sp)
            val_dists_sp = np.array(val_dists_sp)
            train_dists.append(train_dists_sp)
            test_dists.append(test_dists_sp)
            val_dists.append(val_dists_sp)
        # stack across animals
        train_dists = np.stack(train_dists, axis=0)
        test_dists = np.stack(test_dists, axis=0)
        val_dists = np.stack(val_dists, axis=0)
        results[score_method] = {"train_dists": train_dists, "test_dists": test_dists, "val_dists": val_dists}
    return results

def construct_filename(dataset_name="caitlin2dwithoutinertial",
                       arena_size=100,
                       smooth_std=1,
                       num_train_test_splits=10,
                       num_cv_splits=2,
                       train_frac=0.2,
                       val_frac=None,
                  ):

    fname = "sample_shape_alpha_across_animals"
    fname += f"_dataset{dataset_name}"
    fname += f"_arenasize{arena_size}"
    fname += f"_smoothstd_{smooth_std}"
    fname += f"_numtrsp{num_train_test_splits}"
    if val_frac is not None:
        fname += f"_val_frac{val_frac}"
    else:
        fname += f"_numcvsp{num_cv_splits}"
    fname += f"_trainfrac{train_frac}"
    fname += ".npz"
    return fname

def save_results(fit_results,
                 dataset_name="caitlin2dwithoutinertial",
                  smooth_std=1,
                    num_train_test_splits=10,
                    num_cv_splits=2,
                    train_frac=0.2,
                    val_frac=None,
                    **kwargs
                  ):

    save_dir = BASE_DIR_RESULTS
    if dataset_name == "caitlin2dwithoutinertial":
        arena_size = 100
    else:
        assert(dataset_name in ["ofreward_combined", "of_only", "reward_only"])
        arena_size = 150
    print(f"Saving results to this directory {save_dir}")
    fname = construct_filename(dataset_name=dataset_name,
                              arena_size=arena_size,
                              smooth_std=smooth_std,
                                num_train_test_splits=num_train_test_splits,
                                num_cv_splits=num_cv_splits,
                                train_frac=train_frac,
                                val_frac=val_frac,
                  )

    np.savez(os.path.join(save_dir, fname), fit_results)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="caitlin2dwithoutinertial")
    parser.add_argument("--train_frac", type=float, default=0.2)
    parser.add_argument("--val_frac", type=float, default=None)
    parser.add_argument("--num_cv_splits", type=int, default=2)
    args = parser.parse_args()

    print('Looking up params')
    param_lookup = build_param_lookup(dataset_name=args.dataset_name,
                                      train_frac=args.train_frac,
                                      val_frac=args.val_frac,
                                      num_cv_splits=args.num_cv_splits)
    print('NUM JOBS', len(list(param_lookup.keys())))
    curr_params = get_params_from_workernum(worker_num=os.environ.get('SLURM_ARRAY_TASK_ID'), param_lookup=param_lookup)
    print('Curr params', curr_params)
    fit_results = sample_across_neurons(**curr_params)
    save_results(fit_results,
                 **curr_params)
