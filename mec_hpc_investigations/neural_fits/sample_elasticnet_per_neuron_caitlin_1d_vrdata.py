import os, copy
import itertools
import numpy as np
from mec_hpc_investigations.neural_data.remap_utils import aggregate_responses_1dvr
from mec_hpc_investigations.core.utils import get_params_from_workernum
from mec_hpc_investigations.core.default_dirs import CAITLIN1D_VR_INTERANIMAL_CON_RESULTS_ELASTICNET, BASE_DIR_PACKAGED, BASE_DIR_MODELS
from mec_hpc_investigations.neural_fits.utils import generate_train_test_splits
from mec_hpc_investigations.neural_data.metrics import noise_estimation
from mec_hpc_investigations.neural_fits.interanimal_utils import filename_constructor, construct_holdout_sources
from mec_hpc_investigations.core.constants import ALPHA_RNG_1D, L1_RATIO_RNG_1D
from sklearn.model_selection import KFold

def build_param_lookup(metric='pearsonr',
                       num_train_test_splits=10,
                       n_iter=100,
                       train_frac=0.2,
                       correction='spearman_brown_split_half_denominator',
                       agg_resp_kwargs={},
                       first_N_name=None,
                       n_jobs=20):

    spec_resp_agg = aggregate_responses_1dvr(**agg_resp_kwargs)

    # build param lookup
    animals = list(spec_resp_agg.keys())
    param_lookup = {}
    key = 0

    for animal_pair in itertools.permutations(animals, r=2):
        source_animal = animal_pair[0]
        target_animal = animal_pair[1]
        for target_sess, target_sess_maps in spec_resp_agg[target_animal].items():
            for target_sess_map_idx, target_sess_map in enumerate(target_sess_maps):
                for source_sess, source_sess_maps in spec_resp_agg[source_animal].items():
                    for source_sess_map_idx, source_sess_map in enumerate(source_sess_maps):
                        source_N = source_sess_map
                        target_N = target_sess_map

                        param_lookup[str(key)] = {'source_N': source_N,
                                                   'target_N': target_N,
                                                   'source_animal_name': source_animal,
                                                   'source_session_name': source_sess,
                                                   'source_map_name': source_sess_map_idx,
                                                   'target_animal_name': target_animal,
                                                   'target_session_name': target_sess,
                                                   'target_map_name': target_sess_map_idx,
                                                   'data_config': agg_resp_kwargs,
                                                   'metric': metric,
                                                   'num_train_test_splits': num_train_test_splits,
                                                   'n_iter': n_iter,
                                                   'train_frac': train_frac,
                                                   'correction': correction,
                                                   'first_N_name': first_N_name,
                                                   'n_jobs': n_jobs
                                                   }
                        key += 1


    return param_lookup


def compute_interanimal_consistencies(source_N,
                                      target_N,
                                      source_animal_name,
                                      source_session_name,
                                      source_map_name,
                                      target_animal_name,
                                      target_session_name,
                                      target_map_name,
                                      data_config,
                                      metric='pearsonr',
                                      num_train_test_splits=10,
                                      n_iter=100,
                                      train_frac=0.2,
                                      correction='spearman_brown_split_half_denominator',
                                      first_N_name=None,
                                      num_cv_splits=2,
                                      n_jobs=20):

    kf = KFold(n_splits=num_cv_splits)

    assert(len(target_N.shape) == 3)
    if source_N is not None:
        assert(len(source_N.shape) == 3)
        assert(source_N.shape[1] == target_N.shape[1])

    first_N = None
    if first_N_name is not None:
        assert(source_N is not None)
        first_N = np.load(os.path.join(BASE_DIR_MODELS, first_N_name + ".npz"), allow_pickle=True)['arr_0'][()]
        # number of states between first and source should be equal
        if len(first_N.shape) == 2:
            assert(first_N.shape[0] == source_N.shape[1])
        elif len(first_N.shape) == 3:
            assert(first_N.shape[1] == source_N.shape[1])
        else:
            raise ValueError

    train_test_splits = generate_train_test_splits(num_states=target_N.shape[1], # number of position bins
                                                       num_splits=num_train_test_splits,
                                                       train_frac=train_frac)
    agg_results = {}
    for n in range(target_N.shape[-1]):
        curr_target_N = np.expand_dims(target_N[:, :, n], axis=-1)

        results_list = []
        for alpha, l1_ratio in list(itertools.product(ALPHA_RNG_1D, L1_RATIO_RNG_1D)):

            results = {"test_scores": [], "val_scores": [], "alpha": alpha, "l1_ratio": l1_ratio, "alpha_renorm": []}
            for curr_sp in train_test_splits:
                # divide alpha by number of training samples to match alpha of sklearn Ridge
                # see: https://stackoverflow.com/questions/47365978/scikit-learn-elastic-net-approaching-ridge
                curr_alpha = alpha/((float)(len(curr_sp['train'])))
                results["alpha_renorm"].append(curr_alpha)

                try:
                    source_map_kwargs = {"map_type": "sklinear", "map_kwargs": {"regression_type": "ElasticNet",
                                                                                "regression_kwargs": {"alpha": curr_alpha, "l1_ratio": l1_ratio}}}
                    val_results_agg = np.nan
                    for cv_idx, (train_idx, val_idx) in enumerate(kf.split(curr_sp['train'])):
                        curr_val_results = noise_estimation(target_N=curr_target_N,
                                                        source_N=source_N,
                                                        source_map_kwargs=source_map_kwargs,
                                                        first_N=first_N,
                                                        parallelize_per_target_unit=False,
                                                        train_img_idx=curr_sp['train'][train_idx],
                                                        test_img_idx=curr_sp['train'][val_idx],
                                                        metric=metric,
                                                        mode=correction,
                                                        center=np.nanmean,
                                                        summary_center='raw',
                                                        sync=True,
                                                        n_iter=n_iter,
                                                        n_jobs=n_jobs)
                        if cv_idx == 0:
                            val_results_agg = copy.deepcopy(curr_val_results)
                        else:
                            val_results_agg = (val_results_agg + curr_val_results)

                    # average across cv splits
                    val_results_mean = (1.0/num_cv_splits)*val_results_agg
                    results["val_scores"].append(val_results_mean)

                    # fit to entire train set, and eval on test set
                    curr_results = noise_estimation(target_N=curr_target_N,
                                                    source_N=source_N,
                                                    source_map_kwargs=source_map_kwargs,
                                                    first_N=first_N,
                                                    parallelize_per_target_unit=False,
                                                    train_img_idx=curr_sp['train'],
                                                    test_img_idx=curr_sp['test'],
                                                    metric=metric,
                                                    mode=correction,
                                                    center=np.nanmean,
                                                    summary_center='raw',
                                                    sync=True,
                                                    n_iter=n_iter,
                                                    n_jobs=n_jobs)

                    results["test_scores"].append(curr_results)
                except:
                    results["val_scores"] = np.nan
                    results["test_scores"] = np.nan
                    break

            results_list.append(results)

        agg_results[n] = results_list

    filename = filename_constructor(dataset=f"caitlin1dvr_sample_elasticnet_numcv{num_cv_splits}",
                                      source_animal_name=source_animal_name,
                                      source_session_name=source_session_name,
                                      source_map_name=source_map_name,
                                      target_animal_name=target_animal_name,
                                      target_session_name=target_session_name,
                                      target_map_name=target_map_name,
                                      data_config=data_config,
                                      metric=metric,
                                      num_train_test_splits=num_train_test_splits,
                                      n_iter=n_iter,
                                      train_frac=train_frac,
                                      correction=correction,
                                      first_N_name=first_N_name,
                                      file_ext='.npz')
    filename = os.path.join(CAITLIN1D_VR_INTERANIMAL_CON_RESULTS_ELASTICNET, filename)
    np.savez(filename, agg_results)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--first_N_name", type=str, default=None)
    parser.add_argument("--n_jobs", type=int, default=20)
    args = parser.parse_args()

    param_lookup = build_param_lookup(
        metric='pearsonr',
        first_N_name=args.first_N_name,
        n_jobs=args.n_jobs
    )

    print("NUM TOTAL JOBS", len(list(param_lookup.keys())))
    curr_params = get_params_from_workernum(worker_num=os.environ.get('SLURM_ARRAY_TASK_ID'),
                                            param_lookup=param_lookup)
    print("CURR PARAMS", curr_params)
    compute_interanimal_consistencies(**curr_params)
