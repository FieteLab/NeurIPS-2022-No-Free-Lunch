import numpy as np
import os, copy
from mec_hpc_investigations.neural_data.datasets import CaitlinDatasetWithoutInertial
from mec_hpc_investigations.core.default_dirs import BASE_DIR_RESULTS, CAITLIN2D_MODEL_BORDERGRID_RESULTS, CAITLIN2D_INTERANIMAL_CC_MAP_MODEL_RESULTS
from mec_hpc_investigations.models.utils import get_rnn_activations, get_model_gridscores
from mec_hpc_investigations.models.scores import GridScorer
from mec_hpc_investigations.core.constants import gridscore_starts, gridscore_ends
from mec_hpc_investigations.neural_fits.utils import get_max_layer_fits
from mec_hpc_investigations.neural_data.border_score_utils import compute_border_score_solstad

import tensorflow as tf

def get_layer(rnn_type, activation, mode, dataset, eval_arena_size, layer_max):
    curr_model_activations, _, _ = get_rnn_activations(rnn_type=rnn_type, activation=activation, mode=mode,
                                                       dataset=dataset, eval_arena_size=eval_arena_size)
    curr_model_activations = curr_model_activations[layer_max]
    assert(len(curr_model_activations.shape) == 3)
    return curr_model_activations

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=None, required=True)
    parser.add_argument("--rnn_type", type=str, default=None, required=True)
    parser.add_argument("--cv_type", type=str, default="elasticnet_max")
    ARGS = parser.parse_args()

    rnn_type = ARGS.rnn_type
    cv_type = ARGS.cv_type

    # If GPUs available, select which to train on
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=ARGS.gpu

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    print("Loading data")
    eval_arena_size = 100
    train_frac = 0.2
    num_train_test_splits = 10
    num_cv_splits = 2
    suffix = f"{cv_type}_caitlin2darena{eval_arena_size}_trainfrac{train_frac}"
    suffix += f"_numtrsp{num_train_test_splits}_numcvsp{num_cv_splits}"
    suffix += "_fixedinteranimal"

    model_gridscores = np.load(os.path.join(BASE_DIR_RESULTS, f"arena{eval_arena_size}_model_gridscores.npz"), allow_pickle=True)["arr_0"][()]
    model_borderscores = np.load(os.path.join(BASE_DIR_RESULTS, f"arena{eval_arena_size}_model_borderscores.npz"), allow_pickle=True)["arr_0"][()]

    dataset_obj = CaitlinDatasetWithoutInertial()
    dataset_obj.package_data()
    dataset = dataset_obj.packaged_data
    scorer_obj = GridScorer(nbins=20,
               mask_parameters=zip(gridscore_starts, gridscore_ends.tolist()),
               min_max=False)

    results = {}
    for mode in ["random", eval_arena_size, "original", "pos"]:
        for activation in ["linear", "tanh", "relu", "sigmoid"]:
            model_name = f"{rnn_type}_{activation}_{mode}"
            print(f"Current model: {model_name}")
            results[model_name] = {}

            results_dict_alllayers = np.load(os.path.join(CAITLIN2D_INTERANIMAL_CC_MAP_MODEL_RESULTS, f"{model_name}_{suffix}.npz"), allow_pickle=True)["arr_0"][()]
            _, layer_max = get_max_layer_fits(results_dict_alllayers,
                                              eval_arena_size=eval_arena_size)

            results[model_name][layer_max] = {}

            if layer_max == "g":
                model_key = model_name.lower()
                if model_key in model_gridscores.keys():
                    results[model_name][layer_max]["gridscores"] = model_gridscores[model_key]
                else:
                    curr_model_activations = get_layer(rnn_type=rnn_type,
                                                       activation=activation,
                                                       mode=mode,
                                                       dataset=dataset,
                                                       eval_arena_size=eval_arena_size,
                                                       layer_max=layer_max)
                    results[model_name][layer_max]["gridscores"] = get_model_gridscores(scorer=scorer_obj, model_resp=curr_model_activations)

                if model_key in model_borderscores.keys():
                    results[model_name][layer_max]["borderscores"] = model_borderscores[model_key]
                else:
                    curr_model_activations = get_layer(rnn_type=rnn_type,
                                                       activation=activation,
                                                       mode=mode,
                                                       dataset=dataset,
                                                       eval_arena_size=eval_arena_size,
                                                       layer_max=layer_max)
                    results[model_name][layer_max]["borderscores"] = compute_border_score_solstad(np.transpose(curr_model_activations, (2, 0, 1)))
            else:
                curr_model_activations = get_layer(rnn_type=rnn_type,
                                                   activation=activation,
                                                   mode=mode,
                                                   dataset=dataset,
                                                   eval_arena_size=eval_arena_size,
                                                   layer_max=layer_max)

                results[model_name][layer_max]["gridscores"] = get_model_gridscores(scorer=scorer_obj, model_resp=curr_model_activations)
                results[model_name][layer_max]["borderscores"] = compute_border_score_solstad(np.transpose(curr_model_activations, (2, 0, 1)))

    np.savez(os.path.join(CAITLIN2D_MODEL_BORDERGRID_RESULTS, f"{rnn_type}_bestneuralpredlayer_bordergridscores_{suffix}.npz"), results)

