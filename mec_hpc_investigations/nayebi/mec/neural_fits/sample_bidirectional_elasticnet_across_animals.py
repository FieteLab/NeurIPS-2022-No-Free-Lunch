import os, copy
import numpy as np
import itertools
from mec_hpc_investigations.core.default_dirs import BASE_DIR_RESULTS
from mec_hpc_investigations.core.utils import get_params_from_workernum
from mec_hpc_investigations.core.constants import ALPHA_RNG, L1_RATIO_RNG
from mec_hpc_investigations.neural_fits.utils import generate_train_test_splits, prep_data_2d
from mec_hpc_investigations.neural_data.datasets import CaitlinDatasetWithoutInertial, RewardDataset
from mec_hpc_investigations.neural_data.utils import aggregate_responses, concat_resp_conds
from joblib import delayed, Parallel
from sklearn.model_selection import KFold

def build_param_lookup(dataset_name="caitlin2dwithoutinertial",
                       train_frac=0.2,
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
                              "num_cv_splits": num_cv_splits,
                              }
    return param_lookup

def sample_across_neurons(spec_resp_agg,
                        dataset_name="caitlin2dwithoutinertial",
                        num_train_test_splits=10,
                        num_cv_splits=2,
                        train_frac=0.2,
                        alpha_range=ALPHA_RNG,
                        l1_ratio_rng=L1_RATIO_RNG,
                        n_jobs=20):

    results = Parallel(n_jobs=n_jobs)(delayed(_sampler)
                       (spec_resp_agg=spec_resp_agg,
                        num_train_test_splits=num_train_test_splits,
                        num_cv_splits=num_cv_splits,
                        train_frac=train_frac,
                        alpha=alpha,
                        l1_ratio=l1_ratio)
                        for alpha, l1_ratio in list(itertools.product(alpha_range, l1_ratio_range)))

    return results

def _sampler(spec_resp_agg,
            train_frac,
            alpha,
            l1_ratio,
            num_train_test_splits=10,
            num_cv_splits=2):

    kf = KFold(n_splits=num_cv_splits)

    results = {"alpha_renorm": {},
               "alpha": alpha,
               "l1_ratio": l1_ratio,
               "train_frac": train_frac,
               "val_scores": {},
               "val_scores_forward": {},
               "val_scores_backward": {}}

    arena_animals = list(spec_resp_agg.keys())
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
                                                   num_splits=num_train_test_splits,
                                                  )
        results["alpha_renorm"][target_animal] = []
        val_scores_for_sp = []
        val_scores_back_sp = []
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
            results["alpha_renorm"][target_animal].append(curr_alpha)
            curr_map_kwargs = {"regression_type": "ElasticNet",
                               "regression_kwargs": {"alpha": curr_alpha, "l1_ratio": l1_ratio}}
            try:
                # doing our own cv so we don't average over neurons
                val_score_agg_for = np.nan
                val_score_agg_back = np.nan
                for cv_idx, (cv_train_idx, val_idx) in enumerate(kf.split(curr_sp['train'])):
                    M_for = PipelineNeuralMap(map_type="sklinear",
                                              map_kwargs=curr_map_kwargs)
                    M_back = PipelineNeuralMap(map_type="sklinear",
                                              map_kwargs=curr_map_kwargs)
                    X_train_cv = X[cv_train_idx]
                    Y_train_cv = Y[cv_train_idx]
                    X_val = X[val_idx]
                    Y_val = Y[val_idx]
                    M_for.fit(X_train_cv, Y_train_cv)
                    M_back.fit(Y_train_cv, X_train_cv)
                    curr_val_score_for = M_for.score(M_for.predict(X_val), Y_val)
                    curr_val_score_back = M_back.score(M_back.predict(Y_val), X_val)
                    if cv_idx == 0:
                        val_score_agg_for = copy.deepcopy(curr_val_score_for)
                        val_score_agg_back = copy.deepcopy(curr_val_score_back)
                    else:
                        val_score_agg_for = (val_score_agg_for + curr_val_score_for)
                        val_score_agg_back = (val_score_agg_back + curr_val_score_back)

                val_score_for = (1.0/num_cv_splits)*val_score_agg_for
                assert(val_score_for.ndim == 1)
                val_score_for = np.median(val_score_for)
                val_scores_for_sp.append(val_score_for)
                val_score_back = (1.0/num_cv_splits)*val_score_agg_back
                assert(val_score_back.ndim == 1)
                val_score_back = np.median(val_score_back)
                val_scores_back_sp.append(val_score_back)
                # take median across neurons then average
                val_score = 0.5*val_score_for + 0.5*val_score_back
                val_scores_sp.append(val_score)
            except:
                # some choices of hyperparameters may not work so we set this to nan and break in this case
                val_scores_for_sp = np.nan
                val_scores_back_sp = np.nan
                val_scores_sp = np.nan
                break

        results["val_scores_forward"][target_animal] = val_scores_for_sp
        results["val_scores_backward"][target_animal] = val_scores_back_sp
        results["val_scores"][target_animal] = val_scores_sp
    return results

def construct_filename(dataset_name="caitlin2dwithoutinertial",
                       arena_size=100,
                       smooth_std=1,
                       num_train_test_splits=10,
                       num_cv_splits=2,
                       train_frac=0.2,
                  ):

    fname = "sample_bidirectional_elasticnet_across_animals"
    fname += f"_dataset{dataset_name}"
    fname += f"_arenasize{arena_size}"
    fname += f"_smoothstd_{smooth_std}"
    fname += f"_numtrsp{num_train_test_splits}"
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
                  )

    np.savez(os.path.join(save_dir, fname), fit_results)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="caitlin2dwithoutinertial")
    parser.add_argument("--train_frac", type=float, default=0.2)
    parser.add_argument("--num_cv_splits", type=int, default=2)
    args = parser.parse_args()

    print('Looking up params')
    param_lookup = build_param_lookup(dataset_name=args.dataset_name,
                                      train_frac=args.train_frac,
                                      num_cv_splits=args.num_cv_splits)
    print('NUM JOBS', len(list(param_lookup.keys())))
    curr_params = get_params_from_workernum(worker_num=os.environ.get('SLURM_ARRAY_TASK_ID'), param_lookup=param_lookup)
    print('Curr params', curr_params)
    fit_results = sample_across_neurons(**curr_params)
    save_results(fit_results,
                 **curr_params)
