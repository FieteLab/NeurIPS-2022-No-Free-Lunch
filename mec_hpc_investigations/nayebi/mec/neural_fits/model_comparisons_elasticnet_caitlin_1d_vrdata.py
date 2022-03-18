import os
import itertools
import numpy as np
from mec_hpc_investigations.neural_data.remap_utils import aggregate_responses_1dvr
from mec_hpc_investigations.core.utils import get_params_from_workernum
from mec_hpc_investigations.core.default_dirs import BASE_DIR_MODELS, CAITLIN1D_VR_INTERANIMAL_CON_RESULTS_ELASTICNET, CAITLIN1D_VR_INTERANIMAL_CON_RESULTS_ELASTICNET_MAP, CAITLIN1D_VR_INTERANIMAL_CON_RESULTS_ELASTICNET_MAP_MODEL_RESULTS
from mec_hpc_investigations.neural_fits.utils import generate_train_test_splits
from mec_hpc_investigations.neural_data.metrics import noise_estimation
from mec_hpc_investigations.neural_fits.interanimal_utils import filename_constructor

NMF_COMPONENTS = [9, 100, 256, 512]
NMF_MODES = ["env1d", "original"]
SIMPLE_RNNS = ["rnn", "VanillaRNN"]
GATED_RNNS = ["lstm", "UGRNN", "GRU"]
RNN_ACTIVATIONS = ["linear", "tanh", "relu", "sigmoid"]
RNN_TASK_MODES = ["random", "env1d", "original", "pos"]
CUE_RNNS = ["RewardUGRNN2_relu_fixedcue2d_prob0", "RewardUGRNN2_relu_fixedcue2d_prob05", "UGRNN_relu_fixedcue2d_only"]
CONTROLS = ["velocityinput_none_env1d", "cueinput_none_env1d", "placecells_none_env1d"]
MODEL_TYPES = ["controls", "nmf", "simple_rnns", "gated_rnns", "cue_rnns"]

def build_model_paths_1d(model_types=MODEL_TYPES):
    model_paths = {}
    for model_type in model_types:
        if model_type == "controls":
            for model_name in CONTROLS:
                model_paths[model_name] = os.path.join(BASE_DIR_MODELS, model_name + "_activations_1d.npz")
        elif model_type == "nmf":
            for nmf_component in NMF_COMPONENTS:
                for mode in NMF_MODES:
                    model_name = f"{model_type}_{nmf_component}components_{mode}"
                    model_paths[model_name] = os.path.join(BASE_DIR_MODELS, model_name + "_activations_1d.npz")
        elif model_type == "simple_rnns":
            for rnn_type in SIMPLE_RNNS:
                for activation in RNN_ACTIVATIONS:
                    for mode in RNN_TASK_MODES:
                        model_name = f"{rnn_type}_{activation}_{mode}"
                        model_paths[model_name] = os.path.join(BASE_DIR_MODELS, model_name + "_alllayeractivations_1d.npz")
        elif model_type == "gated_rnns":
            for rnn_type in GATED_RNNS:
                for activation in RNN_ACTIVATIONS:
                    for mode in RNN_TASK_MODES:
                        model_name = f"{rnn_type}_{activation}_{mode}"
                        model_paths[model_name] = os.path.join(BASE_DIR_MODELS, model_name + "_alllayeractivations_1d.npz")
        elif model_type == "cue_rnns":
            for model_name in CUE_RNNS:
                model_paths[model_name] = os.path.join(BASE_DIR_MODELS, model_name + "_alllayeractivations_1d.npz")
        else:
            raise ValueError
    return model_paths

def build_param_lookup(model_types=MODEL_TYPES,
                       metric='pearsonr',
                       num_train_test_splits=10,
                       n_iter=100,
                       train_frac=0.2,
                       mode='pairwise',
                       correction='spearman_brown_split_half_denominator',
                       agg_resp_kwargs={},
                       n_jobs=5):

    spec_resp_agg = aggregate_responses_1dvr(**agg_resp_kwargs)

    # build param lookup
    animals = list(spec_resp_agg.keys())
    param_lookup = {}
    key = 0
    # build model list
    model_paths = build_model_paths_1d(model_types=model_types)

    for model_name, model_path in model_paths.items():
        curr_source = np.load(model_path, allow_pickle=True)['arr_0'][()]
        for target_animal in animals:
            for target_sess, target_sess_maps in spec_resp_agg[target_animal].items():
                for target_sess_map_idx, target_sess_map in enumerate(target_sess_maps):
                    param_lookup[str(key)] = {'curr_source': curr_source,
                                               'target_N': target_sess_map,
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

def compute_model_comparisons(curr_source,
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
                              n_jobs=5):

    map_kwargs = np.load(os.path.join(CAITLIN1D_VR_INTERANIMAL_CON_RESULTS_ELASTICNET_MAP, "map_kwargs_per_cell_1d.npz"),
                         allow_pickle=True)["arr_0"][()]
    assert(len(target_N.shape) == 3)
    train_test_splits = generate_train_test_splits(num_states=target_N.shape[1], # number of position bins
                                                       num_splits=num_train_test_splits,
                                                       train_frac=train_frac)
    if not isinstance(curr_source, dict):
        curr_source = {"": curr_source}

    results_dict = {}
    for model_layer, source_N in curr_source.items():
        assert(len(source_N.shape) == 2)
        assert(source_N.shape[0] == target_N.shape[1])

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
        results_dict[model_layer] = agg_results_neurons

    filename = filename_constructor(dataset='caitlin1dvr',
                                      prefix="",
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
    filename = os.path.join(CAITLIN1D_VR_INTERANIMAL_CON_RESULTS_ELASTICNET_MAP_MODEL_RESULTS, filename)
    np.savez(filename, results_dict)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_types", type=str, default=None)
    parser.add_argument("--n_jobs", type=int, default=5)
    parser.add_argument("--missing_fn", type=str, default=None)
    args = parser.parse_args()
    if args.missing_fn is not None:
        param_lookup = np.load(os.path.join(CAITLIN1D_VR_INTERANIMAL_CON_RESULTS_ELASTICNET, args.missing_fn), allow_pickle=True)["arr_0"][()]
    else:
        if args.model_types is None:
            model_types = MODEL_TYPES
        else:
            model_types = args.model_types.split(",")
        print("Looking up params...")
        SHARED_KWARGS = {"model_types": model_types, "n_jobs": args.n_jobs}
        param_lookup = build_param_lookup(**SHARED_KWARGS)

    print("NUM TOTAL JOBS", len(list(param_lookup.keys())))
    curr_params = get_params_from_workernum(worker_num=os.environ.get('SLURM_ARRAY_TASK_ID'),
                                            param_lookup=param_lookup)
    print("CURR PARAMS", curr_params)
    compute_model_comparisons(**curr_params)
