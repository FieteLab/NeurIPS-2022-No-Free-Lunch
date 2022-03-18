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
    parser.add_argument("--metric", type=str, default="grid")
    parser.add_argument("--metric_thresh", type=float, default=None)
    parser.add_argument("--subsample_frac", type=float, default=1.0)
    ARGS = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=ARGS.gpu
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    print(f"Subsample fraction: {ARGS.subsample_frac}")
    print(f"Metric: {ARGS.metric}")

    metric_thresh_list = [ARGS.metric_thresh]

    if ARGS.metric == "grid":
        het_thresh = 0.3
        if ARGS.metric_thresh is None:
            metric_thresh_list = [0.3, 0.55, 0.8, 1.2]
    elif ARGS.metric == "border":
        het_thresh = 0.5
        if ARGS.metric_thresh is None:
            metric_thresh_list = [0.5, 0.6, 0.7, 0.8]
    else:
        raise ValueError

    for curr_metric_thresh in metric_thresh_list:
        print(f"Score threshold: {curr_metric_thresh}")

        thresh_save_nm = str(curr_metric_thresh).replace(".", "")

        dataset_obj = CaitlinDatasetWithoutInertial()
        dataset_obj.package_data()
        dataset = dataset_obj.packaged_data

        _, curr_model, curr_cfg = get_rnn_activations(rnn_type=ARGS.rnn_type,
                                                      activation=ARGS.activation,
                                                      mode=ARGS.mode,
                                                      dataset=dataset, eval_arena_size=100)

        curr_scores = load_cached_metric_scores(metric=ARGS.metric, layer=ARGS.layer,
                                                rnn_type=ARGS.rnn_type,
                                                activation=ARGS.activation,
                                                mode=ARGS.mode)
        error_dict_100 = get_knockout_errs(model=curr_model,
                                            eval_cfg=curr_cfg,
                                            num_units=len(curr_scores),
                                            popa_idx=np.where(curr_scores > curr_metric_thresh)[0],
                                            popb_idx=np.where(curr_scores <= het_thresh)[0], # note: we keep the heterogeneous cell definition fixed
                                            subsample_frac=ARGS.subsample_frac,
                                            dataset=dataset,
                                            arena_size=100,
                                            knockout_layer=ARGS.layer)
        sv_nm = f"{ARGS.rnn_type}_{ARGS.activation}_{ARGS.mode}"
        if ARGS.layer != "g":
            sv_nm += f"_layer{ARGS.layer}"
        sv_nm += f"_arena100_knockouterr_{ARGS.metric}thres_{thresh_save_nm}_frac{ARGS.subsample_frac}.npz"
        np.savez(os.path.join(BASE_DIR_RESULTS, sv_nm), error_dict_100)
