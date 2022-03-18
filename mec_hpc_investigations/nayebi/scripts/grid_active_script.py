import numpy as np
import tensorflow as tf
import os
from mec_hpc_investigations.neural_data.datasets import CaitlinDatasetWithoutInertial
from mec_hpc_investigations.models.utils import configure_options, get_model_activations, load_trained_model, get_model_gridscores, get_mask_errors
from mec_hpc_investigations.models.scores import GridScorer
from mec_hpc_investigations.core.constants import gridscore_starts, gridscore_ends
from mec_hpc_investigations.core.default_dirs import BASE_DIR_RESULTS

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=None, required=True)
    parser.add_argument("--subsample_frac", type=float, default=1.0, required=True)
    parser.add_argument("--grid_thresh", type=float, default=0.3)
    ARGS = parser.parse_args()
    print(f"Subsample fraction: {ARGS.subsample_frac}")
    print(f"Grid score threshold: {ARGS.grid_thresh}")
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=ARGS.gpu

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    thresh_save_nm = str(ARGS.grid_thresh).replace(".", "")

    dataset_obj = CaitlinDatasetWithoutInertial()
    dataset_obj.package_data()
    dataset = dataset_obj.packaged_data

    options = configure_options()

    #lstm_relu_random = load_trained_model(rnn_type="lstm", activation="relu", arena_size=None, random_init=True)
    lstm_relu_original = load_trained_model(rnn_type="lstm", activation="relu", arena_size=None)
    lstm_relu_100_acts = get_model_activations(dataset=dataset,
                                                       model=lstm_relu_original,
                                                       cfg=options,
                                                       arena_size=100,
                                                       n_avg=100,
                                                       trajectory_seed=0)
    #lstm_relu_original_acts = get_model_activations(dataset=dataset,
    #                                                model=lstm_relu_original,
    #                                                   cfg=options,
    #                                                   arena_size="original",
    #                                                   n_avg=100,
    #                                                   trajectory_seed=0)

    #scorer_original = GridScorer(nbins=44,
    #           mask_parameters=zip(gridscore_starts, gridscore_ends.tolist()),
    #           min_max=False)
    #model_scores_original = get_model_gridscores(scorer=scorer_original, model_resp=lstm_relu_original_acts)
    #error_dict_original = get_mask_errors(
    #                               model=lstm_relu_original,
    #                               eval_cfg=options,
    #                               model_scores=model_scores_original,
    #                               subsample_frac=ARGS.subsample_frac,
    #                               grid_thresh=ARGS.grid_thresh,
    #                               arena_size="original")
    #np.savez(os.path.join(BASE_DIR_RESULTS, f"lstm_relu_original_originalarena_modelperformance_gridthres_{thresh_save_nm}_frac{ARGS.subsample_frac}.npz"), error_dict_original)

    scorer_100 = GridScorer(nbins=20,
               mask_parameters=zip(gridscore_starts, gridscore_ends.tolist()),
               min_max=False)
    print("Computing grid scores")
    model_scores_100 = get_model_gridscores(scorer=scorer_100, model_resp=lstm_relu_100_acts)
    print("Computing mask errors")
    error_dict_100 = get_mask_errors(model=lstm_relu_original,
                                     eval_cfg=options,
                                    model_scores=model_scores_100,
                                    subsample_frac=ARGS.subsample_frac,
                                    grid_thresh=ARGS.grid_thresh,
                                    dataset=dataset,
                                    arena_size=100)
    np.savez(os.path.join(BASE_DIR_RESULTS, f"lstm_relu_original_arena100_modelperformance_gridthres_{thresh_save_nm}_frac{ARGS.subsample_frac}.npz"), error_dict_100)
