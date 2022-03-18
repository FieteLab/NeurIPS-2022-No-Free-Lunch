import numpy as np
import os, copy
from mec_hpc_investigations.neural_data.datasets import CaitlinHPCDataset
from mec_hpc_investigations.core.default_dirs import BASE_DIR_MODELS
from mec_hpc_investigations.models.utils import get_rnn_activations
import tensorflow as tf

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=None, required=True)
    parser.add_argument("--rnn_type", type=str, default=None, required=True)
    ARGS = parser.parse_args()

    rnn_type = ARGS.rnn_type

    # If GPUs available, select which to train on
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=ARGS.gpu

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    dataset = CaitlinHPCDataset()
    dataset.package_data()

    for eval_arena_size in [62, 50]:
        for mode in ["random", "original", "pos"]:
            for activation in ["linear", "tanh", "relu", "sigmoid"]:

                curr_model_activations, _, _ = get_rnn_activations(rnn_type=rnn_type, activation=activation, mode=mode,
                                                                   dataset=dataset, eval_arena_size=eval_arena_size)

                np.savez(os.path.join(BASE_DIR_MODELS, f"{rnn_type}_{activation}_{mode}_alllayeractivations_caitlin2darena{eval_arena_size}.npz"), curr_model_activations)
