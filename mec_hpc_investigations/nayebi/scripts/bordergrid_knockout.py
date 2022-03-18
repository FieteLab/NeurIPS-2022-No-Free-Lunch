import numpy as np
import tensorflow as tf
import os, copy
from mec_hpc_investigations.neural_data.datasets import CaitlinDatasetWithoutInertial
from mec_hpc_investigations.models.utils import get_knockout_errs, get_rnn_activations, load_cached_metric_scores
from mec_hpc_investigations.core.default_dirs import BASE_DIR_RESULTS, CAITLIN2D_MODEL_BORDERGRID_RESULTS

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=None, required=True)
    parser.add_argument("--rnn_type", type=str, default="lstm")
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--mode", type=str, default="original")
    parser.add_argument("--layer", type=str, default="g")
    ARGS = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=ARGS.gpu
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    dataset_obj = CaitlinDatasetWithoutInertial()
    dataset_obj.package_data()
    dataset = dataset_obj.packaged_data

    model_borderscores = load_cached_metric_scores(metric="border",
                                                    layer=ARGS.layer,
                                                    rnn_type=ARGS.rnn_type,
                                                    activation=ARGS.activation,
                                                    mode=ARGS.mode)
    model_gridscores = load_cached_metric_scores(metric="grid",
                                                    layer=ARGS.layer,
                                                    rnn_type=ARGS.rnn_type,
                                                    activation=ARGS.activation,
                                                    mode=ARGS.mode)

    curr_model_activations, curr_model, curr_cfg = get_rnn_activations(rnn_type=ARGS.rnn_type,
                                                  activation=ARGS.activation,
                                                  mode=ARGS.mode,
                                                  dataset=dataset, eval_arena_size=100)

    bordergrid_error_dict_100 = get_knockout_errs(model=curr_model,
                                                eval_cfg=curr_cfg,
                                                num_units=curr_model_activations[ARGS.layer].shape[-1],
                                                popa_idx=np.where(np.logical_or((model_borderscores > 0.5), (model_gridscores > 0.3)))[0],
                                                popb_idx=np.where(np.logical_and((model_borderscores <= 0.5), (model_gridscores <= 0.3)))[0],
                                                 dataset=dataset,
                                                arena_size=100,
                                                knockout_layer=ARGS.layer)
    sv_nm = f"{ARGS.rnn_type}_{ARGS.activation}_{ARGS.mode}"
    if ARGS.layer != "g":
        sv_nm += f"_layer{ARGS.layer}"
    sv_nm += "_arena100_knockouterr_borderthres_05_gridthres_03.npz"
    np.savez(os.path.join(BASE_DIR_RESULTS, sv_nm), bordergrid_error_dict_100)

