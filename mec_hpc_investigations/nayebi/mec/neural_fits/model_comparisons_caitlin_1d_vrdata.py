import os
import itertools
import numpy as np
from mec_hpc_investigations.neural_data.remap_utils import aggregate_responses_1dvr
from mec_hpc_investigations.core.utils import get_params_from_workernum
from mec_hpc_investigations.core.default_dirs import BASE_DIR_RESULTS, BASE_DIR_MODELS
from mec_hpc_investigations.neural_fits.utils import generate_train_test_splits
from mec_hpc_investigations.neural_data.metrics import noise_estimation
from mec_hpc_investigations.neural_fits.interanimal_utils import filename_constructor

def build_param_lookup(model_types=["rnn", "lstm", "nmf"],
                       task_modes=["random", "original"],
                       activations=["linear", "tanh", "relu"],
                       nmf_components=[9, 100, 256],
                       source_map_kwargs={},
                       metric='pearsonr',
                       num_train_test_splits=10,
                       n_iter=900,
                       train_frac=0.5,
                       mode='pairwise',
                       correction='spearman_brown_split_half_denominator',
                       agg_resp_kwargs={},
                       n_jobs=5):

    spec_resp_agg = aggregate_responses_1dvr(**agg_resp_kwargs)

    # build param lookup
    animals = list(spec_resp_agg.keys())
    param_lookup = {}
    key = 0

    for model_type in model_types:
        if model_type == "nmf":
            for mode in task_modes:
                if mode != "random": # unsupported for this model class
                    for nmf_component in nmf_components:
                        model_name = f"{model_type}_{nmf_component}components_{mode}"
                        curr_source = np.load(os.path.join(BASE_DIR_MODELS, model_name + "_activations_1d.npz"))['arr_0'][()]

                        for target_animal in animals:
                            for target_sess, target_sess_maps in spec_resp_agg[target_animal].items():
                                for target_sess_map_idx, target_sess_map in enumerate(target_sess_maps):
                                    param_lookup[str(key)] = {'source_N': curr_source,
                                                               'target_N': target_sess_map,
                                                               'source_map_kwargs': source_map_kwargs,
                                                               'source_animal_name': model_name,
                                                               'source_session_name': 0,
                                                               'source_map_name': 0,
                                                               'target_animal_name': target_animal,
                                                               'target_session_name': target_sess,
                                                               'target_map_name': target_sess_map_idx,
                                                               'data_config': agg_resp_kwargs,
                                                               'metric': metric,
                                                               'num_train_test_splits': num_train_test_splits,
                                                               'n_iter': n_iter,
                                                               'train_frac': train_frac,
                                                               'correction': correction,
                                                               'n_jobs': n_jobs
                                                               }
                                    key += 1
        else:
            for mode in task_modes:

                for activation in activations:
                    model_name = f"{model_type}_{activation}_{mode}"
                    curr_source = np.load(os.path.join(BASE_DIR_MODELS, model_name + "_activations_1d.npz"))['arr_0'][()]

                    for target_animal in animals:
                        for target_sess, target_sess_maps in spec_resp_agg[target_animal].items():
                            for target_sess_map_idx, target_sess_map in enumerate(target_sess_maps):
                                param_lookup[str(key)] = {'source_N': curr_source,
                                                           'target_N': target_sess_map,
                                                           'source_map_kwargs': source_map_kwargs,
                                                           'source_animal_name': model_name,
                                                           'source_session_name': 0,
                                                           'source_map_name': 0,
                                                           'target_animal_name': target_animal,
                                                           'target_session_name': target_sess,
                                                           'target_map_name': target_sess_map_idx,
                                                           'data_config': agg_resp_kwargs,
                                                           'metric': metric,
                                                           'num_train_test_splits': num_train_test_splits,
                                                           'n_iter': n_iter,
                                                           'train_frac': train_frac,
                                                           'correction': correction,
                                                           'n_jobs': n_jobs
                                                           }
                                key += 1

    return param_lookup

def compute_model_comparisons(source_N,
                              target_N,
                              source_map_kwargs,
                              source_animal_name,
                              source_session_name,
                              source_map_name,
                              target_animal_name,
                              target_session_name,
                              target_map_name,
                              data_config,
                              metric='pearsonr',
                              num_train_test_splits=10,
                              n_iter=900,
                              train_frac=0.5,
                              correction='spearman_brown_split_half_denominator',
                              n_jobs=5):

    assert(len(target_N.shape) == 3)
    assert(source_map_kwargs is not None)
    assert(len(source_N.shape) == 2)
    assert(source_N.shape[0] == target_N.shape[1])

    agg_results = []
    train_test_splits = generate_train_test_splits(num_states=target_N.shape[1], # number of position bins
                                                       num_splits=num_train_test_splits,
                                                       train_frac=train_frac)
    for curr_sp in train_test_splits:
        curr_results = noise_estimation(target_N=target_N,
                                        source_N=source_N,
                                        source_map_kwargs=source_map_kwargs,
                                        parallelize_per_target_unit=False,
                                        train_img_idx=curr_sp['train'], test_img_idx=curr_sp['test'],
                                        metric=metric,
                                        mode=correction,
                                        center=np.nanmean,
                                        summary_center='raw',
                                        sync=True,
                                        n_iter=n_iter,
                                        n_jobs=n_jobs)

        curr_results = np.expand_dims(curr_results, axis=0)
        agg_results.append(curr_results)
    agg_results = np.concatenate(agg_results, axis=0) # (num_train_test_splits, num_bs_trials, num_target_units)

    filename = filename_constructor(dataset='caitlin1dvr',
                                      prefix="",
                                      source_map_kwargs=source_map_kwargs,
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
                                      file_ext='.npz')
    filename = os.path.join(BASE_DIR_RESULTS, filename)
    np.savez(filename, agg_results)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_types", type=str, default="rnn,lstm,nmf")
    parser.add_argument("--task_modes", type=str, default="random,original")
    parser.add_argument("--activations", type=str, default="linear,tanh,relu")
    parser.add_argument("--nmf_components", type=str, default="9,100,256")
    parser.add_argument("--map_type", type=str, default="percentile")
    parser.add_argument("--percentile", type=float, default=0)
    parser.add_argument("--percentile_identity", type=bool, default=False)
    parser.add_argument("--pls_n_components", type=int, default=9)
    parser.add_argument("--pls_fit_per_target_unit", type=bool, default=False)
    parser.add_argument("--n_jobs", type=int, default=5)
    args = parser.parse_args()

    print("Looking up params...")
    SHARED_KWARGS = {"n_jobs": args.n_jobs,
                     "model_types": args.model_types.split(","),
                     "task_modes": args.task_modes.split(","),
                     "activations": args.activations.split(","),
                     "nmf_components": args.nmf_components.split(",")}

    if args.map_type == "percentile":
        source_map_kwargs = {'map_type': 'percentile', 'map_kwargs': {'identity': args.percentile_identity, 'percentile': args.percentile}}
        param_lookup = build_param_lookup(
            source_map_kwargs=source_map_kwargs,
            metric='pearsonr',
            **SHARED_KWARGS
        )
    elif args.map_type == "pls":
        source_map_kwargs = {'map_type': 'pls', 'map_kwargs': {'n_components': args.pls_n_components, 'fit_per_target_unit': args.pls_fit_per_target_unit}}
        param_lookup = build_param_lookup(
            source_map_kwargs=source_map_kwargs,
            metric='pearsonr',
            **SHARED_KWARGS
        )
    elif args.map_type == "rsa":
        source_map_kwargs = {'map_type': 'identity', 'map_kwargs': {}}
        param_lookup = build_param_lookup(
            source_map_kwargs=source_map_kwargs,
            metric='rsa',
            **SHARED_KWARGS
        )
    else:
        raise ValueError(f"{args.map_type} not implemented yet.")

    print("NUM TOTAL JOBS", len(list(param_lookup.keys())))
    curr_params = get_params_from_workernum(worker_num=os.environ.get('SLURM_ARRAY_TASK_ID'),
                                            param_lookup=param_lookup)
    print("CURR PARAMS", curr_params)
    compute_model_comparisons(**curr_params)
