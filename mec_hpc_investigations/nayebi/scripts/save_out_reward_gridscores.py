import numpy as np
import tensorflow as tf
import os
from mec_hpc_investigations.neural_data.datasets import RewardDataset
from mec_hpc_investigations.models.utils import load_trained_model, configure_options, get_model_activations, get_model_gridscores
from mec_hpc_investigations.core.constants import gridscore_starts, gridscore_ends
from mec_hpc_investigations.models.scores import GridScorer
from mec_hpc_investigations.core.default_dirs import BASE_DIR_RESULTS

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=None, required=True)
    parser.add_argument("--eval_mode", type=str, default=None, required=True)
    ARGS = parser.parse_args()
    eval_mode = ARGS.eval_mode
    print("Eval mode", eval_mode)

    # If GPUs available, select which to train on
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=ARGS.gpu
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    of_dataset = RewardDataset(dataset="free_foraging")
    of_dataset.package_data()

    scorer_150 = GridScorer(nbins=len(of_dataset.arena_x_bins)-1,
               mask_parameters=zip(gridscore_starts, gridscore_ends.tolist()),
               min_max=False)

    # make cfgs
    REWARD_ZONE_KWARGS = {"reward_zone_size": 0.2,
                          "reward_zone_min_x": 0.65,
                          "reward_zone_max_x": 0.85,
                          "reward_zone_min_y": 0.65,
                          "reward_zone_max_y": 0.85,
                          "reward_zone_as_input": False}

    no_rz_cfg = configure_options(arena_size=150)

    # center of the 1.5 m x 1.5 m environment, vary number of timesteps to reward zone
    # navigate 7 timesteps to get to reward
    nav7_fixed_rz_per_episode_cfg = configure_options(arena_size=150,
                                                 reward_zone_prob=1.0,
                                                 reward_zone_navigate_timesteps=7,
                                                 **REWARD_ZONE_KWARGS)

    print("Loading trained models")
    ugrnn_relu_original_no_rz = load_trained_model(rnn_type="UGRNN", activation="relu", arena_size=None)

    ugrnn_relu_original_nav7_prob625_rz = load_trained_model(rnn_type="UGRNN", activation="relu",
                                                            arena_size=None,
                                                            run_ID="UGRNN_relu_bigarena_rz_center_exp7",
                                                       reward_zone_prob=0.625,
                                                       reward_zone_navigate_timesteps=7,
                                                       **REWARD_ZONE_KWARGS)

    if eval_mode == "of":
        chosen_cfg = no_rz_cfg
    elif eval_mode == "task":
        chosen_cfg = nav7_fixed_rz_per_episode_cfg
    else:
        raise ValueError

    print("Getting model activations")
    ugrnn_relu_original_acts = get_model_activations(dataset=of_dataset,
                                                    model=ugrnn_relu_original_no_rz,
                                                       cfg=chosen_cfg,
                                                       arena_size=150,
                                                       n_avg=100,
                                                       trajectory_seed=0)

    ugrnn_relu_original_nav7_prob625_acts = get_model_activations(dataset=of_dataset,
                                                    model=ugrnn_relu_original_nav7_prob625_rz,
                                                       cfg=chosen_cfg,
                                                       arena_size=150,
                                                       n_avg=100,
                                                       trajectory_seed=0)

    print("Computing original gridscores")
    model_scores_150 = get_model_gridscores(scorer=scorer_150, model_resp=ugrnn_relu_original_acts)
    np.savez(os.path.join(BASE_DIR_RESULTS, f"ugrnn_relu_original_evalmode{eval_mode}_arena150_gridscores.npz"), model_scores_150)

    print("Computing prob625 grid scores")
    model_scores_prob625_150 = get_model_gridscores(scorer=scorer_150, model_resp=ugrnn_relu_original_nav7_prob625_acts)
    np.savez(os.path.join(BASE_DIR_RESULTS, f"ugrnn_relu_original_nav7_prob625evalmode{eval_mode}_arena150_gridscores.npz"), model_scores_prob625_150)
