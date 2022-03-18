import numpy as np
import os, copy
from mec_hpc_investigations.neural_data.datasets import CaitlinDatasetWithoutInertial
from mec_hpc_investigations.neural_data.border_score_utils import compute_border_score_solstad
from mec_hpc_investigations.core.default_dirs import BASE_DIR_RESULTS, CAITLIN2D_MODEL_BORDERGRID_RESULTS
from mec_hpc_investigations.models.utils import get_rnn_activations, get_model_performance, get_knockout_mask, get_baseline_knockout_mask
import tensorflow as tf
from tqdm import tqdm

def load_cached_gridscores(layer, rnn_type, activation, mode, eval_arena_size, cv_type="elasticnet_max"):
    if layer == "g":
        curr_scores = np.load(os.path.join(BASE_DIR_RESULTS, f"arena{eval_arena_size}_model_gridscores.npz"), allow_pickle=True)['arr_0'][()][f"{rnn_type.lower()}_{activation}_{mode}"]
    else:
        #TODO: extend to other settings if we use a different neural mapping strategy
        train_frac = 0.2
        num_train_test_splits = 10
        num_cv_splits = 2
        suffix = f"{cv_type}_caitlin2darena{eval_arena_size}_trainfrac{train_frac}"
        suffix += f"_numtrsp{num_train_test_splits}_numcvsp{num_cv_splits}"
        suffix += "_fixedinteranimal"
        curr_scores = np.load(os.path.join(CAITLIN2D_MODEL_BORDERGRID_RESULTS,
                              f"{rnn_type}_bestneuralpredlayer_bordergridscores_{suffix}.npz"), allow_pickle=True)['arr_0'][()][f"{rnn_type}_{activation}_{mode}"][layer]['gridscores']
    return curr_scores

def get_curr_model_knockout_performance(knockout_type,
                                curr_knockout_idxs,
                                layer,
                                num_units,
                                base_kwargs,
                               curr_model_activations=None):
    # copy does not work so have to do it iteratively
    curr_kwargs = {}
    for k,v in base_kwargs.items():
        curr_kwargs[k] = v
    mask = get_knockout_mask(num_units=num_units,
                             knockout_idx=curr_knockout_idxs)
    if knockout_type == "zeros":
        curr_kwargs[f"{layer}_mask"] = mask
        _, curr_model_performance = get_model_performance(**curr_kwargs)
    elif knockout_type == "baseline":
        assert(curr_model_activations is not None)
        baseline_knockout = np.mean(curr_model_activations, axis=(0, 1))
        mask_add = get_baseline_knockout_mask(num_units=num_units,
                                              knockout_idx=curr_knockout_idxs,
                                              baseline_values=baseline_knockout)
        curr_kwargs[f"{layer}_mask"] = mask
        curr_kwargs[f"{layer}_mask_add"] = mask_add
        _, curr_model_performance = get_model_performance(**curr_kwargs)
    else:
        raise ValueError
    return curr_model_performance

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=None, required=True)
    parser.add_argument("--rnn_type", type=str, default=None, required=True)
    parser.add_argument("--activation", type=str, default=None, required=True)
    parser.add_argument("--mode", type=str, default=None, required=True)
    parser.add_argument("--layer", type=str, default=None, required=True)
    parser.add_argument("--metric", type=str, default=None, required=True, choices=["grid", "border", "both", "random"])
    parser.add_argument("--knockout_type", type=str, default=None, required=True, choices=["zeros", "baseline"])
    parser.add_argument("--knockout_first", type=str, default="max", choices=["min", "max"])
    parser.add_argument("--num_points", type=int, default=None)
    parser.add_argument("--num_ss", type=int, default=20)
    ARGS = parser.parse_args()

    rnn_type = ARGS.rnn_type
    activation = ARGS.activation
    mode = ARGS.mode
    layer = ARGS.layer
    metric = ARGS.metric
    knockout_type = ARGS.knockout_type
    knockout_first = ARGS.knockout_first if metric != "random" else None
    num_points = ARGS.num_points
    num_ss = ARGS.num_ss

    # If GPUs available, select which to train on
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=ARGS.gpu

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    print("Loading data")
    eval_arena_size = 100
    dataset_obj = CaitlinDatasetWithoutInertial()
    dataset_obj.package_data()
    dataset = dataset_obj.packaged_data

    print("Loading model")
    curr_model_activations, curr_model, curr_cfg = get_rnn_activations(rnn_type=rnn_type, activation=activation, mode=mode,
                                                                       dataset=dataset, eval_arena_size=eval_arena_size)

    curr_model_activations = curr_model_activations[layer]
    #TODO: deal with env1d
    assert(len(curr_model_activations.shape) == 3)
    num_units = curr_model_activations.shape[-1]

    if metric == "both":
        curr_grid_scores = load_cached_gridscores(layer=layer, rnn_type=rnn_type, activation=activation,
                                             mode=mode, eval_arena_size=eval_arena_size,
                                             cv_type="elasticnet_max")
        curr_border_scores = compute_border_score_solstad(np.transpose(curr_model_activations, (2, 0, 1)))
        if knockout_first == "max":
            grid_knockout_order = np.argsort(curr_grid_scores)[::-1]
            border_knockout_order = np.argsort(curr_border_scores)[::-1]
        elif knockout_first == "min":
            grid_knockout_order = np.argsort(curr_grid_scores)
            border_knockout_order = np.argsort(curr_border_scores)
        else:
            raise ValueError
        assert(len(grid_knockout_order) == len(border_knockout_order))
        assert(len(grid_knockout_order) == num_units)
    elif metric != "random":
        if metric == "grid":
            curr_scores = load_cached_gridscores(layer=layer, rnn_type=rnn_type, activation=activation,
                                                 mode=mode, eval_arena_size=eval_arena_size,
                                                 cv_type="elasticnet_max")
        elif metric == "border":
            curr_scores = compute_border_score_solstad(np.transpose(curr_model_activations, (2, 0, 1)))
        else:
            raise ValueError

        if knockout_first == "max":
            knockout_order = np.argsort(curr_scores)[::-1]
        elif knockout_first == "min":
            knockout_order = np.argsort(curr_scores)
        else:
            raise ValueError

        assert(len(knockout_order) == num_units)


    if num_points is None:
        num_points = (num_units+1)

    knockout_performances = {}
    base_kwargs = {"model": curr_model, "cfg": curr_cfg, "dataset": dataset, "arena_size": eval_arena_size}
    for num_knockout_units in tqdm(np.linspace(start=0, stop=num_units, num=num_points, endpoint=True, dtype=int)):
        if num_knockout_units == 0:
            _, curr_model_performance = get_model_performance(**base_kwargs)
        else:
            if metric == "random":
                if num_knockout_units < num_units:
                    curr_model_performance = []
                    for s in range(num_ss):
                        np.random.seed(s)
                        # randomly sample num_knockout_units, num_ss times
                        curr_knockout_idxs = np.random.permutation(num_units)[:num_knockout_units]
                        curr_iter_model_performance = get_curr_model_knockout_performance(knockout_type=knockout_type,
                                                                                    curr_knockout_idxs=curr_knockout_idxs,
                                                                                    layer=layer,
                                                                                    num_units=num_units,
                                                                                    base_kwargs=base_kwargs,
                                                                                    curr_model_activations=curr_model_activations)
                        curr_model_performance.append(curr_iter_model_performance)
                    curr_model_performance = np.stack(curr_model_performance, axis=0)
                else: # ablate all units, so only need to do this once
                    curr_model_performance = get_curr_model_knockout_performance(knockout_type=knockout_type,
                                                                                curr_knockout_idxs=np.arange(num_units),
                                                                                layer=layer,
                                                                                num_units=num_units,
                                                                                base_kwargs=base_kwargs,
                                                                                curr_model_activations=curr_model_activations)
            else:
                if metric == "both":
                    grid_knockout_idxs = grid_knockout_order[:num_knockout_units]
                    border_knockout_idxs = border_knockout_order[:num_knockout_units]
                    curr_knockout_idxs = np.unique(np.concatenate([grid_knockout_idxs, border_knockout_idxs], axis=-1))
                    assert(len(curr_knockout_idxs.shape) == 1)
                else:
                    curr_knockout_idxs = knockout_order[:num_knockout_units]

                curr_model_performance = get_curr_model_knockout_performance(knockout_type=knockout_type,
                                                                            curr_knockout_idxs=curr_knockout_idxs,
                                                                            layer=layer,
                                                                            num_units=num_units,
                                                                            base_kwargs=base_kwargs,
                                                                            curr_model_activations=curr_model_activations)

        if metric == "both":
            # this can be different than the actual knockout number just because it is the unique union of both populations
            if num_knockout_units > 0:
                bg_num = len(curr_knockout_idxs)
            else:
                assert(num_knockout_units == 0)
                bg_num = 0
            knockout_performances[num_knockout_units] = {bg_num: curr_model_performance}
        else:
            knockout_performances[num_knockout_units] = curr_model_performance

    sv_nm = f"{rnn_type}_{activation}_{mode}_sorted_knockout{knockout_type}_{knockout_first}_layer{layer}_metric{metric}"
    if ARGS.num_points is not None:
        sv_nm += f"_numpt{num_points}"
    if metric == "random":
        sv_nm += f"_numss{num_ss}"
    sv_nm += f"_caitlin2darena{eval_arena_size}.npz"
    np.savez(os.path.join(BASE_DIR_RESULTS, sv_nm), knockout_performances)
