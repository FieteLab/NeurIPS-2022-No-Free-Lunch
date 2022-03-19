import os, copy
import numpy as np
import itertools
from mec_hpc_investigations.core.constants import ALPHA_RNG_SHORT, L1_RATIO_RNG_SHORT
from mec_hpc_investigations.core.default_dirs import CAITLIN2D_INTERANIMAL_SAMPLE_RESULTS, OFREWARD_COMBINED_INTERANIMAL_SAMPLE_RESULTS
from mec_hpc_investigations.core.utils import get_params_from_workernum, iterate_dicts
from mec_hpc_investigations.neural_fits.utils import generate_train_test_splits, prep_data_2d, package_scores
from mec_hpc_investigations.neural_mappers.cross_validator import CrossValidator
from mec_hpc_investigations.neural_mappers.pipeline_neural_map import PipelineNeuralMap
from mec_hpc_investigations.neural_data.datasets import CaitlinDatasetWithoutInertial, RewardDataset
from mec_hpc_investigations.neural_data.utils import aggregate_responses, concat_resp_conds
from joblib import delayed, Parallel
from sklearn.model_selection import KFold

TRAIN_FRAC_RNG = list(np.linspace(0.015, 0.5, num=25, endpoint=True))

def build_param_lookup(dataset_name="caitlin2dwithoutinertial",
                       train_frac_range=None,
                       val_frac=None,
                       smooth_std=1,
                       num_cv_splits=3,
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
                              "train_frac_range": train_frac_range,
                              "val_frac": val_frac,
                              "num_cv_splits": num_cv_splits,
                              }
    return param_lookup

def sample_across_neurons(spec_resp_agg,
                        dataset_name="caitlin2dwithoutinertial",
                        num_train_test_splits=10,
                        num_cv_splits=3,
                        train_frac_range=None,
                        val_frac=None,
                        alpha_range=ALPHA_RNG_SHORT,
                        l1_ratio_range=L1_RATIO_RNG_SHORT,
                        n_jobs=5):

    if train_frac_range is None:
        train_frac_range = TRAIN_FRAC_RNG

    results = Parallel(n_jobs=n_jobs)(delayed(_sampler)
                       (spec_resp_agg=spec_resp_agg,
                        num_train_test_splits=num_train_test_splits,
                        num_cv_splits=num_cv_splits,
                        train_frac=train_frac,
                        val_frac=val_frac,
                        alpha=alpha,
                        l1_ratio=l1_ratio)
                        for train_frac, alpha, l1_ratio in list(itertools.product(train_frac_range, alpha_range, l1_ratio_range)))

    return results

def _sampler(spec_resp_agg,
            train_frac,
            alpha,
            l1_ratio,
            num_train_test_splits=10,
            num_cv_splits=3,
            val_frac=None,
            neural_map_str="sklinear"):

    kf = KFold(n_splits=num_cv_splits)

    results = {"alpha_renorm": [],
               "alpha": alpha,
               "l1_ratio": l1_ratio,
               "train_frac": train_frac}

    if val_frac is not None:
        results["val_frac"] = val_frac

    arena_animals = list(spec_resp_agg.keys())
    train_scores = []
    test_scores = []
    val_scores = []
    for target_animal in arena_animals:
        curr_source_animals = list(set(arena_animals) - set([target_animal]))
        assert(target_animal not in curr_source_animals)
        assert(len(curr_source_animals) == len(arena_animals) - 1)

        source_animal_resp = np.concatenate([spec_resp_agg[source_animal]["resp"] for source_animal in curr_source_animals], axis=-1)
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

        train_scores_sp = []
        test_scores_sp = []
        val_scores_sp = []
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
            curr_map_kwargs = {"regression_type": "ElasticNet",
                               "regression_kwargs": {"alpha": curr_alpha, "l1_ratio": l1_ratio}}
            M = PipelineNeuralMap(map_type=neural_map_str,
                                  map_kwargs=curr_map_kwargs)
            if val_frac is not None:
                val_idx = curr_sp["val"]
                X_val = X[val_idx]
                Y_val = Y[val_idx]
                try:
                    M.fit(X_train, Y_train)
                    Y_train_pred = M.predict(X_train)
                    train_score = M.score(Y_train_pred, Y_train)
                    Y_val_pred = M.predict(X_val)
                    val_score = M.score(Y_val_pred, Y_val)
                    Y_test_pred = M.predict(X_test)
                    test_score = M.score(Y_test_pred, Y_test)

                    train_scores_sp.append(train_score)
                    test_scores_sp.append(test_score)
                    val_scores_sp.append(val_score)
                except:
                    # some choices of hyperparameters may not work so we set this to nan and break in this case
                    nan_pred = (np.zeros(Y.shape[-1]) + np.nan)
                    train_scores_sp.append(nan_pred)
                    test_scores_sp.append(nan_pred)
                    val_scores_sp.append(nan_pred)
            else:
                try:
                    # doing our own cv so we don't average over neurons
                    val_score_agg = np.nan
                    for cv_idx, (cv_train_idx, val_idx) in enumerate(kf.split(curr_sp['train'])):
                        M = PipelineNeuralMap(map_type=neural_map_str,
                                              map_kwargs=curr_map_kwargs)
                        X_train_cv = X[cv_train_idx]
                        Y_train_cv = Y[cv_train_idx]
                        X_val = X[val_idx]
                        Y_val = Y[val_idx]
                        M.fit(X_train_cv, Y_train_cv)
                        curr_val_score = M.score(M.predict(X_val), Y_val)
                        if cv_idx == 0:
                            val_score_agg = copy.deepcopy(curr_val_score)
                        else:
                            val_score_agg = (val_score_agg + curr_val_score)

                    val_score = (1.0/num_cv_splits)*val_score_agg
                    M = PipelineNeuralMap(map_type=neural_map_str,
                                          map_kwargs=curr_map_kwargs)
                    M.fit(X_train, Y_train)
                    Y_train_pred = M.predict(X_train)
                    train_score = M.score(Y_train_pred, Y_train)
                    Y_test_pred = M.predict(X_test)
                    test_score = M.score(Y_test_pred, Y_test)

                    train_scores_sp.append(train_score)
                    test_scores_sp.append(test_score)
                    val_scores_sp.append(val_score)
                except:
                    # some choices of hyperparameters may not work so we set this to nan and break in this case
                    nan_pred = (np.zeros(Y.shape[-1]) + np.nan)
                    train_scores_sp.append(nan_pred)
                    test_scores_sp.append(nan_pred)
                    val_scores_sp.append(nan_pred)

        # mean across train/test splits
        train_scores_sp = np.nanmean(np.stack(train_scores_sp, axis=0), axis=0)
        test_scores_sp = np.nanmean(np.stack(test_scores_sp, axis=0), axis=0)
        val_scores_sp = np.nanmean(np.stack(val_scores_sp, axis=0), axis=0)
        train_scores.append(train_scores_sp)
        test_scores.append(test_scores_sp)
        val_scores.append(val_scores_sp)
    # concatenate and take median across neurons
    train_scores = np.concatenate(train_scores, axis=-1)
    test_scores = np.concatenate(test_scores, axis=-1)
    val_scores = np.concatenate(val_scores, axis=-1)
    results["train_scores"] = np.nanmedian(train_scores, axis=-1)
    results["test_scores"] = np.nanmedian(test_scores, axis=-1)
    results["val_scores"] = np.nanmedian(val_scores, axis=-1)
    return results

def construct_filename(dataset_name="caitlin2dwithoutinertial",
                       arena_size=100,
                       smooth_std=1,
                       num_train_test_splits=10,
                       num_cv_splits=3,
                       neural_map_str="sklinear",
                       train_frac_range=None,
                       val_frac=None,
                  ):

    fname = "sample_across_neurons"
    fname += f"_dataset{dataset_name}"
    fname += f"_arenasize{arena_size}"
    fname += f"_smoothstd_{smooth_std}"
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
                 dataset_name="caitlin2dwithoutinertial",
                  smooth_std=1,
                    num_train_test_splits=10,
                    num_cv_splits=3,
                    neural_map_str="sklinear",
                    train_frac_range=None,
                    val_frac=None,
                    **kwargs
                  ):

    if dataset_name == "caitlin2dwithoutinertial":
        save_dir = CAITLIN2D_INTERANIMAL_SAMPLE_RESULTS
        arena_size = 100
    else:
        assert(dataset_name in ["ofreward_combined", "of_only", "reward_only"])
        save_dir = OFREWARD_COMBINED_INTERANIMAL_SAMPLE_RESULTS
        arena_size = 150
    print(f"Saving results to this directory {save_dir}")
    fname = construct_filename(dataset_name=dataset_name,
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
    args = parser.parse_args()

    if args.train_frac_range is not None:
        train_frac_range = [(float)(tr_e) for tr_e in args.train_frac_range.split(",")]
    else:
        train_frac_range = None
    print('Looking up params')
    param_lookup = build_param_lookup(dataset_name=args.dataset_name,
                                      train_frac_range=train_frac_range,
                                      val_frac=args.val_frac,
                                      num_cv_splits=args.num_cv_splits)
    print('NUM JOBS', len(list(param_lookup.keys())))
    curr_params = get_params_from_workernum(worker_num=os.environ.get('SLURM_ARRAY_TASK_ID'), param_lookup=param_lookup)
    print('Curr params', curr_params)
    fit_results = sample_across_neurons(**curr_params)
    save_results(fit_results,
                 **curr_params)
