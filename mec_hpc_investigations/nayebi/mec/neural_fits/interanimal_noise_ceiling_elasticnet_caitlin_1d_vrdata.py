import os, copy
import itertools
import numpy as np
from mec_hpc_investigations.neural_data.remap_utils import aggregate_responses_1dvr
from mec_hpc_investigations.core.utils import get_params_from_workernum
from mec_hpc_investigations.core.default_dirs import CAITLIN1D_VR_INTERANIMAL_CON_RESULTS_ELASTICNET_MAP_INTERANIMAL_RESULTS, CAITLIN1D_VR_INTERANIMAL_CON_RESULTS_ELASTICNET_MAP, BASE_DIR_MODELS
from mec_hpc_investigations.neural_fits.utils import generate_train_test_splits
from mec_hpc_investigations.neural_data.metrics import noise_estimation
from mec_hpc_investigations.neural_fits.interanimal_utils import filename_constructor, construct_holdout_sources

def build_param_lookup(metric='pearsonr',
                       num_train_test_splits=10,
                       n_iter=100,
                       train_frac=0.2,
                       correction='spearman_brown_split_half_denominator',
                       agg_resp_kwargs={},
                       first_N_name=None,
                       n_jobs=5):

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
                                      n_jobs=5):

    map_kwargs = np.load(os.path.join(CAITLIN1D_VR_INTERANIMAL_CON_RESULTS_ELASTICNET_MAP, "map_kwargs_per_cell_1d.npz"),
                         allow_pickle=True)["arr_0"][()]

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
    agg_results_neurons = []
    for curr_n in range(target_N.shape[-1]):
        curr_target_N = np.expand_dims(target_N[:, :, curr_n], axis=-1)
        agg_results = []
        for curr_sp_idx, curr_sp in enumerate(train_test_splits):
            curr_map_kwargs = map_kwargs[target_animal_name][target_session_name][curr_n][curr_sp_idx]
            source_map_kwargs = {"map_type": "sklinear",
                                 "map_kwargs": curr_map_kwargs}
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

            curr_results = np.expand_dims(curr_results, axis=0)
            agg_results.append(curr_results)
        agg_results = np.concatenate(agg_results, axis=0) # (num_train_test_splits, num_bs_trials, 1)
        agg_results_neurons.append(agg_results)
    # (num_train_test_splits, num_bs_trials, num_target_neurons)
    agg_results_neurons = np.concatenate(agg_results_neurons, axis=-1)


    filename = filename_constructor(dataset="caitlin1dvr",
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
    filename = os.path.join(CAITLIN1D_VR_INTERANIMAL_CON_RESULTS_ELASTICNET_MAP_INTERANIMAL_RESULTS, filename)
    np.savez(filename, agg_results_neurons)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--first_N_name", type=str, default=None)
    parser.add_argument("--n_jobs", type=int, default=5)
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
