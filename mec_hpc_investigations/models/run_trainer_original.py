import tensorflow as tf
import os
from mec_hpc_investigations.models.utils import configure_options, configure_model
from mec_hpc_investigations.models.trainer import Trainer
from mec_hpc_investigations.core.default_dirs import BANINO_REP_DIR

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=None, required=True)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--rnn_type", type=str, default="RNN")
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--arena_size", type=float, default=None)
    parser.add_argument("--cue_input_only", type=bool, default=False)
    parser.add_argument("--cue_2d_input", type=str, default=None)
    parser.add_argument("--cue_2d_prob", type=float, default=1.0)
    parser.add_argument("--env_1d", type=bool, default=False)
    parser.add_argument("--const_velocity_1d", type=bool, default=False)
    parser.add_argument("--cue_input_mode_1d", type=str, default=None)
    parser.add_argument("--place_cell_identity", type=bool, default=False)
    parser.add_argument("--place_cell_predict", type=bool, default=False)
    parser.add_argument("--num_pc_pred", type=int, default=512)
    parser.add_argument("--pc_k", type=int, default=None)
    parser.add_argument("--pc_activation", type=str, default="relu")
    parser.add_argument("--pc_rnn_func", type=str, default=None)
    parser.add_argument("--pc_rnn_initial_state", type=bool, default=False)
    parser.add_argument("--reward_zone_size", type=float, default=None)
    parser.add_argument("--reward_zone_prob", type=float, default=None)
    parser.add_argument("--reward_zone_min_x", type=float, default=None)
    parser.add_argument("--reward_zone_max_x", type=float, default=None)
    parser.add_argument("--reward_zone_min_y", type=float, default=None)
    parser.add_argument("--reward_zone_max_y", type=float, default=None)
    parser.add_argument("--exclude_reward_zone_as_input", type=bool, default=False)
    parser.add_argument("--reward_zone_navigate_timesteps", type=int, default=None)
    parser.add_argument("--run_ID", type=str, default=None)
    ARGS = parser.parse_args()

    # If GPUs available, select which to train on
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=ARGS.gpu

    if ARGS.save_dir == "banino":
        save_dir = BANINO_REP_DIR
    else:
        save_dir = ARGS.save_dir

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Training options and hyperparameters
    if ARGS.exclude_reward_zone_as_input:
        reward_zone_as_input = False
    else:
        reward_zone_as_input = True

    cue_2d_input_kwargs = None
    if ARGS.cue_2d_input is not None:
        if ARGS.cue_2d_input == "fixed_extents":
            CUE_1 = {"center_x": 0.0, "center_y": 0.0, "width": 0.3, "height": 0.3}
            CUE_2 = {"center_x": 0.35, "center_y": 0.35, "width": 0.2, "height": 0.2}
            CUE_3 = {"center_x": -0.3, "center_y": -0.3, "width": 0.1, "height": 0.1}
            CUE_4 = {"center_x": 0.6, "center_y": 0.5, "width": 0.1, "height": 0.1}
            CUE_5 = {"center_x": -0.7, "center_y": -0.6, "width": 0.06, "height": 0.06}
            cue_2d_input_kwargs = {"cue_extents": [CUE_1, CUE_2, CUE_3, CUE_4, CUE_5], "cue_prob": ARGS.cue_2d_prob}
        else:
            cue_2d_input_kwargs = {}
    options = configure_options(save_dir=save_dir,
                                rnn_type=ARGS.rnn_type,
                                activation=ARGS.activation,
                                arena_size=ARGS.arena_size,
                                env_1d=ARGS.env_1d,
                                const_velocity_1d=ARGS.const_velocity_1d,
                                cue_input_mode_1d=ARGS.cue_input_mode_1d,
                                place_cell_identity=ARGS.place_cell_identity,
                                place_cell_predict=ARGS.place_cell_predict,
                                num_pc_pred=ARGS.num_pc_pred,
                                pc_k=ARGS.pc_k,
                                pc_activation=ARGS.pc_activation,
                                pc_rnn_func=ARGS.pc_rnn_func,
                                pc_rnn_initial_state=ARGS.pc_rnn_initial_state,
                                cue_2d_input_kwargs=cue_2d_input_kwargs,
                                cue_input_only=ARGS.cue_input_only,
                                run_ID=ARGS.run_ID,
                                reward_zone_size=ARGS.reward_zone_size,
                                reward_zone_prob=ARGS.reward_zone_prob,
                                reward_zone_min_x=ARGS.reward_zone_min_x,
                                reward_zone_max_x=ARGS.reward_zone_max_x,
                                reward_zone_min_y=ARGS.reward_zone_min_y,
                                reward_zone_max_y=ARGS.reward_zone_max_y,
                                reward_zone_as_input=reward_zone_as_input,
                                reward_zone_navigate_timesteps=ARGS.reward_zone_navigate_timesteps)

    model = configure_model(options, rnn_type=ARGS.rnn_type)
    trainer = Trainer(options, model)
    trainer.train(n_epochs=options.n_epochs,
                  n_grad_steps_per_epoch=options.n_steps,
                  save=True)

