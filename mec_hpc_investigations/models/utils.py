import scipy
import os
import numpy as np
import copy
import tensorflow as tf
from tqdm import tqdm
import pickle

from mec_hpc_investigations.models.helper_classes import Options, PlaceCells, HeadDirectionCells
from mec_hpc_investigations.core.utils import dict_to_str
from mec_hpc_investigations.models.visualize import compute_ratemaps, plot_ratemaps
from mec_hpc_investigations.neural_data.utils import get_xy_bins
from mec_hpc_investigations.core.default_dirs import BASE_DIR_MODELS
from mec_hpc_investigations.models.model import RNN, LSTM, UGRNN, VanillaRNN, GRU
from mec_hpc_investigations.models.model import RewardUGRNN2, RewardRNN, RewardLSTM, RewardLSTM2
from mec_hpc_investigations.models.model import LSTMPCDense, LSTMPCRNN
from mec_hpc_investigations.models.model import BaninoRNN
from mec_hpc_investigations.models.trajectory_generator import TrajectoryGenerator
from mec_hpc_investigations.neural_data.utils import get_position_bins_1d


def set_env_dims(options: Options):
    if options.min_x is None or options.max_x is None:
        assert options.box_width_in_m is not None
        options.min_x = -options.box_width_in_m / 2.0
        options.max_x = options.box_width_in_m / 2.0

    if options.min_y is None or options.max_y is None:
        assert options.box_height_in_m is not None
        options.min_y = -options.box_height_in_m / 2.0
        options.max_y = options.box_height_in_m / 2.0


def generate_run_ID(options):
    '''
    Create a unique run ID from the most relevant
    parameters. Remaining parameters can be found in
    params.npy file.
    '''
    params = [
        'steps', str(options.sequence_length),
        'batch', str(options.batch_size),
        options.rnn_type,
        str(options.Ng),
        options.activation,
        'rf', str(options.place_cell_rf),
        'DoG', str(options.DoG),
        'periodic', str(options.is_periodic),
        'lr', str(options.learning_rate),
        'weight_decay', str(options.weight_decay),
        'min_x', str(options.min_x),
        'max_x', str(options.max_x),
        'min_y', str(options.min_y),
        'max_y', str(options.max_y),
    ]
    if options.place_cell_identity:
        params.extend(['place_id', str(options.place_cell_identity)])
    if options.place_cell_predict:
        params.extend(['num_pc_pred', str(options.num_pc_pred)])
        params.extend(['pc_k', str(options.pc_k)])
        params.extend(['pc_activation', str(options.pc_activation)])
        if options.pc_rnn_func is not None:
            params.extend(['pc_rnn_func', str(options.pc_rnn_func)])
        if options.pc_rnn_initial_state:
            params.extend(['pc_init', str(options.pc_rnn_initial_state)])
    if hasattr(options, 'cue_2d_input_kwargs') and (options.cue_2d_input_kwargs is not None):
        params.extend(['ci2d', dict_to_str(options.cue_2d_input_kwargs)])
    if hasattr(options, 'const_velocity_1d') and options.const_velocity_1d:
        params.append('cv1d')
    if hasattr(options, 'cue_input_mode_1d') and (options.cue_input_mode_1d is not None):
        params.extend(['ci1d', options.cue_input_mode_1d])
    if hasattr(options, 'cue_input_only') and (options.cue_input_only):
        params.append('cionly')
    if hasattr(options, 'reward_zone_size'):
        params.extend(['rz_size', str(options.reward_zone_size)])
        params.extend(['rz_prob', str(options.reward_zone_prob)])
        params.extend(['rz_x_offset', str(options.reward_zone_x_offset)])
        params.extend(['rz_y_offset', str(options.reward_zone_y_offset)])
        params.extend(['rz_min_x', str(options.reward_zone_min_x)])
        params.extend(['rz_max_x', str(options.reward_zone_max_x)])
        params.extend(['rz_min_y', str(options.reward_zone_min_y)])
        params.extend(['rz_max_y', str(options.reward_zone_max_y)])
        if not options.reward_zone_as_input:
            params.extend(['rz_i', str(options.reward_zone_as_input)])
        if options.reward_zone_navigate_timesteps is not None:
            params.extend(['rz_nvt', str(options.reward_zone_navigate_timesteps)])
    if options.banino_place_cell:
        params.append('bpc')
    if options.optimizer_class.lower() != "adam":
        params.append(options.optimizer_class.lower())
    if options.clipvalue is not None:
        params.extend(['clpv', str(options.clipvalue)])
    if options.Np != 512:  # default value
        params.extend(['Np', str(options.Np)])
    if hasattr(options, "Nhdc"):
        params.extend(['Nhdc', str(options.Nhdc)])
        params.extend(['hdconc', str(options.hdc_concentration)])
    if options.rnn_type.lower() == "baninornn":
        params.extend(['brnn', options.banino_rnn_type])
        params.extend(['brnunits', str(options.banino_rnn_nunits)])
        params.extend(['brdp', str(options.banino_dropout_rate)])

    separator = '_'
    run_ID = separator.join(params)
    run_ID = run_ID.replace('.', '')

    return run_ID


def get_model_gridscores(scorer, model_resp):
    model_scores = []
    for cell_idx in range(model_resp.shape[-1]):
        curr_grid_score = scorer.get_scores(model_resp[:, :, cell_idx])[0]
        model_scores.append(curr_grid_score)
    return np.array(model_scores)


def err_subroutine(model, eval_cfg, grid_idx, nongrid_idx, num_units, dataset=None, arena_size=100):
    grid_mask = np.zeros(num_units)
    grid_mask[grid_idx] = 1
    grid_mask = tf.constant(grid_mask, dtype=tf.float32)

    nongrid_mask = np.zeros(num_units)
    nongrid_mask[nongrid_idx] = 1
    nongrid_mask = tf.constant(nongrid_mask, dtype=tf.float32)

    _, model_grid_mask_errs = get_model_performance(model=model,
                                                    cfg=eval_cfg,
                                                    dataset=dataset,
                                                    arena_size=arena_size,
                                                    g_mask=grid_mask)

    _, model_nongrid_mask_errs = get_model_performance(model=model,
                                                       cfg=eval_cfg,
                                                       dataset=dataset,
                                                       arena_size=arena_size,
                                                       g_mask=nongrid_mask)

    error_dict = {"grid": model_grid_mask_errs, "nongrid": model_nongrid_mask_errs}
    return error_dict


def get_mask_errors(model, model_scores, eval_cfg,
                    subsample_frac=1.0,
                    grid_thresh=0.3, dataset=None,
                    arena_size=100, num_samples=100):
    """ Get errors of model when keep active a certain population of units."""
    grid_idx = np.where(model_scores > grid_thresh)[0]
    nongrid_idx = np.where(model_scores <= grid_thresh)[0]

    if (num_samples is None) or (len(grid_idx) == len(nongrid_idx)):
        error_dict = err_subroutine(model=model,
                                    eval_cfg=eval_cfg,
                                    grid_idx=grid_idx,
                                    nongrid_idx=nongrid_idx,
                                    dataset=dataset,
                                    num_units=len(model_scores),
                                    arena_size=arena_size)
    else:
        if len(nongrid_idx) > len(grid_idx):
            error_dict = {"grid": [], "nongrid": []}
            for s in range(num_samples):
                np.random.seed(s)
                # subsample same number of nongrid cells as grid cells
                ss_len = (int)(np.ceil(subsample_frac * len(grid_idx)))
                print(f"Subsample frac: {subsample_frac}, len: {ss_len}")
                nongrid_idx_ss = np.random.permutation(nongrid_idx)[:ss_len]
                grid_idx_ss = np.random.permutation(grid_idx)[:ss_len]
                curr_error_dict = err_subroutine(model=model,
                                                 eval_cfg=eval_cfg,
                                                 grid_idx=grid_idx_ss,
                                                 nongrid_idx=nongrid_idx_ss,
                                                 dataset=dataset,
                                                 num_units=len(model_scores),
                                                 arena_size=arena_size)

                error_dict["grid"].extend(curr_error_dict["grid"])
                error_dict["nongrid"].extend(curr_error_dict["nongrid"])

        elif len(grid_idx) > len(nongrid_idx):
            error_dict = {"grid": [], "nongrid": []}
            for s in range(num_samples):
                np.random.seed(s)
                # subsample same number of grid cells as nongrid cells
                ss_len = (int)(np.ceil(subsample_frac * len(nongrid_idx)))
                print(f"Subsample frac: {subsample_frac}, len: {ss_len}")
                grid_idx_ss = np.random.permutation(grid_idx)[:ss_len]
                nongrid_idx_ss = np.random.permutation(nongrid_idx)[:ss_len]
                curr_error_dict = err_subroutine(model=model,
                                                 eval_cfg=eval_cfg,
                                                 grid_idx=grid_idx_ss,
                                                 nongrid_idx=nongrid_idx_ss,
                                                 dataset=dataset,
                                                 num_units=len(model_scores),
                                                 arena_size=arena_size)

                error_dict["grid"].extend(curr_error_dict["grid"])
                error_dict["nongrid"].extend(curr_error_dict["nongrid"])

    return error_dict


def norm_pred(model_err, base_err):
    return np.divide((model_err - base_err), base_err)


def get_knockout_mask(num_units, knockout_idx):
    """this mask will be multiplicative"""
    knockout_mask = np.ones(num_units)
    knockout_mask[knockout_idx] = 0
    knockout_mask = tf.constant(knockout_mask, dtype=tf.float32)
    return knockout_mask


def get_baseline_knockout_mask(num_units, knockout_idx, baseline_values=None):
    """this mask will be additive"""
    knockout_mask = None
    if baseline_values is not None:
        assert (len(baseline_values.shape) == 1)
        assert (len(baseline_values) == num_units)
        knockout_mask = np.zeros(num_units)
        knockout_mask[knockout_idx] = baseline_values[knockout_idx]
        knockout_mask = tf.constant(knockout_mask, dtype=tf.float32)
    return knockout_mask


def err_subroutine_knockout(model, eval_cfg, popa_idx, popb_idx, num_units,
                            dataset=None, arena_size=100, baseline_values=None,
                            knockout_layer="g"):
    popa_knockout = get_knockout_mask(num_units=num_units, knockout_idx=popa_idx)
    popa_knockout_baseline = get_baseline_knockout_mask(num_units=num_units,
                                                        knockout_idx=popa_idx,
                                                        baseline_values=baseline_values)

    popb_knockout = get_knockout_mask(num_units=num_units, knockout_idx=popb_idx)
    popb_knockout_baseline = get_baseline_knockout_mask(num_units=num_units,
                                                        knockout_idx=popb_idx,
                                                        baseline_values=baseline_values)

    popa_mask_kwargs = {f"{knockout_layer}_mask": popa_knockout, f"{knockout_layer}_mask_add": popa_knockout_baseline}
    _, model_popa_knockout_errs = get_model_performance(model=model,
                                                        cfg=eval_cfg,
                                                        dataset=dataset,
                                                        arena_size=arena_size,
                                                        **popa_mask_kwargs)

    popb_mask_kwargs = {f"{knockout_layer}_mask": popb_knockout, f"{knockout_layer}_mask_add": popb_knockout_baseline}
    _, model_popb_knockout_errs = get_model_performance(model=model,
                                                        cfg=eval_cfg,
                                                        dataset=dataset,
                                                        arena_size=arena_size,
                                                        **popb_mask_kwargs)

    error_dict = {"popa_knockout": model_popa_knockout_errs, "popb_knockout": model_popb_knockout_errs}
    return error_dict


def get_knockout_errs(model, eval_cfg,
                      popa_idx, popb_idx,
                      num_units,
                      subsample_frac=1.0,
                      dataset=None,
                      arena_size=100, num_samples=100,
                      baseline_values=None,
                      knockout_layer="g"):
    """ Get errors of model when knocking out populations of units in equal amounts."""

    if (num_samples is None) or (len(popa_idx) == len(popb_idx)):
        error_dict = err_subroutine_knockout(model=model,
                                             eval_cfg=eval_cfg,
                                             popa_idx=popa_idx,
                                             popb_idx=popb_idx,
                                             dataset=dataset,
                                             num_units=num_units,
                                             arena_size=arena_size,
                                             baseline_values=baseline_values,
                                             knockout_layer=knockout_layer)
    else:
        if len(popb_idx) > len(popa_idx):
            error_dict = {"popa_knockout": [], "popb_knockout": []}
            for s in range(num_samples):
                np.random.seed(s)
                # subsample same number of popb cells as popa cells
                ss_len = (int)(np.ceil(subsample_frac * len(popa_idx)))
                print(f"Subsample frac: {subsample_frac}, len: {ss_len}")
                popb_idx_ss = np.random.permutation(popb_idx)[:ss_len]
                popa_idx_ss = np.random.permutation(popa_idx)[:ss_len]
                curr_error_dict = err_subroutine_knockout(model=model,
                                                          eval_cfg=eval_cfg,
                                                          popa_idx=popa_idx_ss,
                                                          popb_idx=popb_idx_ss,
                                                          dataset=dataset,
                                                          num_units=num_units,
                                                          arena_size=arena_size,
                                                          baseline_values=baseline_values,
                                                          knockout_layer=knockout_layer)

                error_dict["popa_knockout"].extend(curr_error_dict["popa_knockout"])
                error_dict["popb_knockout"].extend(curr_error_dict["popb_knockout"])

        elif len(popa_idx) > len(popb_idx):
            error_dict = {"popa_knockout": [], "popb_knockout": []}
            for s in range(num_samples):
                np.random.seed(s)
                # subsample same number of popa cells as popb cells
                ss_len = (int)(np.ceil(subsample_frac * len(popb_idx)))
                print(f"Subsample frac: {subsample_frac}, len: {ss_len}")
                popa_idx_ss = np.random.permutation(popa_idx)[:ss_len]
                popb_idx_ss = np.random.permutation(popb_idx)[:ss_len]
                curr_error_dict = err_subroutine_knockout(model=model,
                                                          eval_cfg=eval_cfg,
                                                          popa_idx=popa_idx_ss,
                                                          popb_idx=popb_idx_ss,
                                                          dataset=dataset,
                                                          num_units=num_units,
                                                          arena_size=arena_size,
                                                          baseline_values=baseline_values,
                                                          knockout_layer=knockout_layer)

                error_dict["popa_knockout"].extend(curr_error_dict["popa_knockout"])
                error_dict["popb_knockout"].extend(curr_error_dict["popb_knockout"])

    return error_dict


def calc_err():
    inputs, _, pos = next(gen)
    pred = model(inputs)
    pred_pos = place_cells.get_nearest_cell_pos(pred)
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum((pos - pred_pos) ** 2, axis=-1)))


def compute_variance(res, n_avg, options):
    activations, rate_map, g, p, pos, inp = compute_ratemaps(model, options, res=res, n_avg=n_avg)

    counts = np.zeros([res, res])
    variance = np.zeros([res, res])

    x_all = (pos[:, 0] - options.min_x) / (options.max_x - options.min_x) * res
    y_all = (pos[:, 1] - options.min_y) / (options.max_y - options.min_y) * res
    for i in tqdm(range(len(g))):
        x = int(x_all[i])
        y = int(y_all[i])
        if x >= 0 and x < res and y >= 0 and y < res:
            counts[x, y] += 1
            variance[x, y] += np.linalg.norm(g[i] - activations[:, x, y]) / np.linalg.norm(g[i]) / np.linalg.norm(
                activations[:, x, y])

    for x in range(res):
        for y in range(res):
            if counts[x, y] > 0:
                variance[x, y] /= counts[x, y]

    return variance


def load_trained_weights(model, trainer, weight_dir):
    ''' Load weights stored as a .npy file (for github)'''

    # Train for a single step to initialize weights
    trainer.train(n_epochs=1, n_steps=1, save=False)

    # Load weights from npy array
    weights = np.load(weight_dir, allow_pickle=True)
    model.set_weights(weights)
    print('Loaded trained weights.')


def set_cfg_from_data(dataset, cfg, return_bins=False,
                      arena_size=None, nbins_max=20,
                      meters_convert=True):
    """Sets up an environment based on the parameters from the data."""
    if hasattr(dataset, "arena_x_bins") and hasattr(dataset, "arena_y_bins"):
        arena_x_bins = copy.deepcopy(dataset.arena_x_bins)
        arena_y_bins = copy.deepcopy(dataset.arena_y_bins)
    elif hasattr(dataset, "get_arena_bins"):
        arena_x_bins, arena_y_bins = dataset.get_arena_bins(arena_size)
    else:
        if arena_size is None:
            curr_arena_dataset = dataset
        else:
            curr_arena_dataset = dataset[dataset["arena_size_cm"] == arena_size]
        arena_x_bins, arena_y_bins = get_xy_bins(curr_arena_dataset, nbins_max=nbins_max)

    if meters_convert:
        arena_x_bins /= 100.0
        arena_y_bins /= 100.0
    env_cfg = copy.deepcopy(cfg)
    # in meters rather than cm
    env_cfg.min_x = arena_x_bins[0]
    env_cfg.max_x = arena_x_bins[-1]
    env_cfg.min_y = arena_y_bins[0]
    env_cfg.max_y = arena_y_bins[-1]
    if return_bins:
        return env_cfg, arena_x_bins, arena_y_bins
    else:
        return env_cfg


def get_model_activations(dataset, model, cfg,
                          num_stimuli_types=1,
                          arena_size=None,
                          n_avg=100,
                          nbins_max=20,
                          model_pred_layer="g",
                          trajectory_seed=0,
                          eval_batch_size=None,
                          return_all=False):
    """Returns (num_x_bins, num_y_bins, num_units) model activations for a given arena size"""
    if arena_size == "original":
        eval_cfg = cfg
        assert (eval_cfg.box_width_in_m == eval_cfg.box_height_in_m)
        assert (eval_cfg.box_width_in_m == 2.2)
        # 5 cm bins
        nbins = (int)((eval_cfg.box_width_in_m * 100) / 5.0)
        print(f"Binning into {nbins} bins for big arena")
        arena_x_bins = np.linspace(eval_cfg.min_x, eval_cfg.max_x, nbins + 1, endpoint=True)
        arena_y_bins = np.linspace(eval_cfg.min_y, eval_cfg.max_y, nbins + 1, endpoint=True)
    else:
        # create testing environment
        assert dataset is not None
        assert not isinstance(arena_size, str)
        eval_cfg, arena_x_bins, arena_y_bins = set_cfg_from_data(dataset=dataset,
                                                                 cfg=cfg,
                                                                 arena_size=arena_size,
                                                                 nbins_max=nbins_max,
                                                                 return_bins=True,
                                                                 meters_convert=True)

    print("Using this cfg to compute model activations: ", vars(eval_cfg))

    # get model activations
    _, _, g_dict, p, pos, inp = compute_ratemaps(model,
                                                 options=eval_cfg,
                                                 n_avg=n_avg,
                                                 trajectory_seed=trajectory_seed,
                                                 eval_batch_size=eval_batch_size,
                                                 return_all=True if ((return_all) or (
                                                             model_pred_layer not in ["g", "pc"])) else False)

    if return_all:
        # return all layers of an rnn simultaneously
        assert num_stimuli_types == 1
        assert isinstance(g_dict, dict)
        model_activations = {}
        for k, v in g_dict.items():
            curr_activations = scipy.stats.binned_statistic_2d(x=pos[:, 0],
                                                               y=pos[:, 1],
                                                               values=v.T,
                                                               statistic='mean',
                                                               bins=[arena_x_bins, arena_y_bins])[0]
            model_activations[k] = np.transpose(curr_activations, (1, 2, 0))
    elif hasattr(model, model_pred_layer):
        if model_pred_layer == "pc":
            curr_values = p.T
        elif isinstance(g_dict, dict):
            curr_values = g_dict[model_pred_layer].T
        else:
            curr_values = g_dict.T

        model_activations = scipy.stats.binned_statistic_2d(x=pos[:, 0],
                                                            y=pos[:, 1],
                                                            values=curr_values,
                                                            statistic='mean',
                                                            bins=[arena_x_bins, arena_y_bins])[0]
        # put cells in last dimension (num_x_bins, num_y_bins, num_grid_cells)
        model_activations = np.transpose(model_activations, (1, 2, 0))
    elif model in ["cue_input", "velocity_input"]:
        if model == "cue_input":
            assert (hasattr(eval_cfg, "cue_input_only") and eval_cfg.cue_input_only)
        else:
            if hasattr(eval_cfg, "cue_input_only"):
                assert (eval_cfg.cue_input_only is False)

        model_activations = scipy.stats.binned_statistic_2d(x=pos[:, 0],
                                                            y=pos[:, 1],
                                                            values=inp.T,
                                                            statistic='mean',
                                                            bins=[arena_x_bins, arena_y_bins])[0]
        # (num_x_bins, num_y_bins, input_dim)
        model_activations = np.transpose(model_activations, (1, 2, 0))
    else:
        p_activations = scipy.stats.binned_statistic_2d(x=pos[:, 0],
                                                        y=pos[:, 1],
                                                        values=p.T,
                                                        statistic='mean',
                                                        bins=[arena_x_bins, arena_y_bins])[0]
        p_activations = np.transpose(p_activations, (1, 2, 0))
        if model == "place_cells":
            # (num_x_bins, num_y_bins, num_place_cells)
            model_activations = p_activations
        else:
            assert (hasattr(model, "transform"))
            # (num_x_bins*num_y_bins, num_place_cells)
            p_activations = p_activations.reshape((-1, eval_cfg.Np))
            # (num_x_bins*num_y_bins, num_components)
            model_activations = model.transform(p_activations)

    if num_stimuli_types > 1:
        assert not return_all
        # if we have multiple conditions to compare against
        num_units = model_activations.shape[-1]
        model_activations_flat = model_activations.reshape((-1, num_units))
        concat_model_activations = np.concatenate([model_activations_flat for _ in range(num_stimuli_types)], axis=0)
        return concat_model_activations
    else:
        return model_activations


def get_model_performance(model, cfg,
                          eval_batch_size=None,
                          dataset=None,
                          arena_size="original",
                          n_avg=100,
                          nbins_max=20,
                          env_1d=False,
                          trajectory_seed=0,
                          **kwargs):
    if (arena_size == "original") and (not env_1d):
        eval_cfg = cfg
        assert (eval_cfg.box_width_in_m == eval_cfg.box_height_in_m)
        assert (eval_cfg.box_width_in_m == 2.2)
    else:
        # create testing environment
        # the calls to these functions are more conveniences rather than configuring
        # your own cfg and passing it in
        if env_1d:
            eval_cfg, pos_bins = set_cfg_from_vr1ddata(dataset=dataset,
                                                       cfg=cfg,
                                                       return_bins=True,
                                                       meters_convert=True)
        else:
            assert dataset is not None
            assert not isinstance(arena_size, str)
            eval_cfg, arena_x_bins, arena_y_bins = set_cfg_from_data(dataset=dataset,
                                                                     cfg=cfg,
                                                                     arena_size=arena_size,
                                                                     nbins_max=nbins_max,
                                                                     return_bins=True,
                                                                     meters_convert=True)

    print(f"Using this cfg to evaluate performance: {vars(eval_cfg)}")

    if hasattr(eval_cfg, "Nhdc"):
        trajectory_generator = TrajectoryGenerator(options=eval_cfg,
                                                   place_cells=PlaceCells(eval_cfg),
                                                   head_direction_cells=HeadDirectionCells(eval_cfg),
                                                   trajectory_seed=trajectory_seed)
    else:
        trajectory_generator = TrajectoryGenerator(options=eval_cfg,
                                                   place_cells=PlaceCells(eval_cfg),
                                                   trajectory_seed=trajectory_seed)

    if not n_avg:
        n_avg = 1000 // eval_cfg.sequence_length

    losses = []
    errs = []
    for index in tqdm(range(n_avg), leave=False, desc='Computing performance'):
        inputs, pos_batch, p_batch = trajectory_generator.get_batch(batch_size=eval_batch_size)
        loss_batch, err_batch = model.compute_loss(inputs,
                                                   p_batch,
                                                   pos=pos_batch,
                                                   **kwargs)
        losses.append(loss_batch)
        errs.append(err_batch)

    losses = np.array(losses)
    errs = np.array(errs)

    return losses, errs


def set_cfg_from_vr1ddata(dataset, cfg, return_bins=False,
                          nbins=80,
                          meters_convert=True):
    """Sets up an environment based on the parameters from the data."""
    if (dataset is None) and (nbins == 80):
        # much faster
        print("Loading caitlin 1d vr data cached bins")
        from mec_hpc_investigations.core.default_dirs import BASE_DIR_PACKAGED
        pos_bins = np.load(os.path.join(BASE_DIR_PACKAGED, "caitlin1dvr_pos_bins_80.npz"))["arr_0"][()]
    else:
        print("Loading bins from data")
        pos_bins = get_position_bins_1d(dataset, nbins=nbins)

    if meters_convert:
        pos_bins /= 100.0
    env_cfg = copy.deepcopy(cfg)
    env_cfg.vr1d = True
    env_cfg.is_periodic = True
    # in meters rather than cm
    env_cfg.min_x = pos_bins[0]
    env_cfg.max_x = pos_bins[-1]
    env_cfg.min_y = 0
    env_cfg.max_y = 0
    if return_bins:
        return env_cfg, pos_bins
    else:
        return env_cfg


def get_model_activations_1d(dataset, model, cfg,
                             num_stimuli_types=1,
                             n_avg=100,
                             nbins=80,
                             model_pred_layer="g",
                             trajectory_seed=0,
                             eval_batch_size=None,
                             return_all=False):
    """Returns (num_pos_bins, num_units) model activations for a given arena size"""
    # create testing environment
    eval_cfg, pos_bins = set_cfg_from_vr1ddata(dataset=dataset,
                                               cfg=cfg,
                                               nbins=nbins,
                                               return_bins=True,
                                               meters_convert=True)

    print("Using this cfg to compute model activations: ", vars(eval_cfg))

    # get model activations
    _, _, g_dict, p, pos, inp = compute_ratemaps(model,
                                                 options=eval_cfg,
                                                 n_avg=n_avg,
                                                 trajectory_seed=trajectory_seed,
                                                 eval_batch_size=eval_batch_size,
                                                 return_all=True if ((return_all) or (
                                                             model_pred_layer not in ["g", "pc"])) else False)

    if return_all:
        # return all layers of an rnn simultaneously
        assert num_stimuli_types == 1
        assert isinstance(g_dict, dict)
        model_activations = {}
        for k, v in g_dict.items():
            curr_activations = scipy.stats.binned_statistic(x=pos[:, 0],
                                                            values=v.T,
                                                            statistic='mean',
                                                            bins=pos_bins)[0]
            model_activations[k] = np.transpose(curr_activations, (1, 0))
    elif hasattr(model, model_pred_layer):
        if model_pred_layer == "pc":
            curr_values = p.T
        elif isinstance(g_dict, dict):
            curr_values = g_dict[model_pred_layer].T
        else:
            curr_values = g_dict.T

        model_activations = scipy.stats.binned_statistic(x=pos[:, 0],
                                                         values=curr_values,
                                                         statistic='mean',
                                                         bins=pos_bins)[0]
        # put cells in last dimension (num_x_bins, num_grid_cells)
        model_activations = np.transpose(model_activations, (1, 0))
    elif model in ["cue_input", "velocity_input"]:
        if model == "cue_input":
            assert (hasattr(eval_cfg, "cue_input_only") and eval_cfg.cue_input_only)
        else:
            if hasattr(eval_cfg, "cue_input_only"):
                assert (eval_cfg.cue_input_only is False)

        model_activations = scipy.stats.binned_statistic(x=pos[:, 0],
                                                         values=inp.T,
                                                         statistic='mean',
                                                         bins=pos_bins)[0]
        # (num_x_bins, input_dim)
        model_activations = np.transpose(model_activations, (1, 0))
    else:
        p_activations = scipy.stats.binned_statistic(x=pos[:, 0],
                                                     values=p.T,
                                                     statistic='mean',
                                                     bins=pos_bins)[0]
        p_activations = np.transpose(p_activations, (1, 0))
        if model == "place_cells":
            model_activations = p_activations
        else:
            assert (hasattr(model, "transform"))
            # (num_x_bins, num_components)
            model_activations = model.transform(p_activations)
    return model_activations


def get_rnn_activations(rnn_type, activation, mode,
                        eval_arena_size, dataset=None,
                        eval_batch_size=None,
                        n_avg=100,
                        **kwargs):
    if eval_arena_size not in ["env1d", "original"]:
        # since in 1d environment it is cached to avoid loading large file
        assert (dataset is not None)
    curr_model_kwargs = {"rnn_type": rnn_type,
                         "activation": activation,
                         "arena_size": mode if isinstance(mode, int) else None,
                         "random_init": True if mode == "random" else False,
                         "dataset": dataset,
                         "place_cell_identity": True if mode == "pos" else False,
                         "ckpt_file": None}
    curr_model_kwargs.update(kwargs)
    if mode == "env1d":
        curr_model_kwargs["env_1d"] = True
    if ((rnn_type == "rnn") and (activation == "linear") and (mode == "pos")):
        curr_model_kwargs["ckpt_file"] = "ckpt-4"
    elif ((rnn_type == "VanillaRNN") and (activation == "linear") and (mode == "pos")):
        curr_model_kwargs["ckpt_file"] = "ckpt-5"
    curr_model = load_trained_model(**curr_model_kwargs)
    curr_cfg_kwargs = copy.deepcopy(curr_model_kwargs)
    if eval_arena_size != "env1d":
        curr_cfg_kwargs["arena_size"] = None if eval_arena_size == "original" else eval_arena_size
    curr_cfg_kwargs.pop("random_init")
    curr_cfg_kwargs.pop("ckpt_file")
    curr_cfg = configure_options(**curr_cfg_kwargs)
    if eval_arena_size == "env1d":
        curr_model_activations = get_model_activations_1d(dataset=dataset,
                                                          model=curr_model,
                                                          cfg=curr_cfg,
                                                          eval_batch_size=eval_batch_size,
                                                          n_avg=n_avg,
                                                          return_all=True)
    else:
        curr_model_activations = get_model_activations(dataset=dataset,
                                                       model=curr_model,
                                                       cfg=curr_cfg,
                                                       arena_size=eval_arena_size,
                                                       eval_batch_size=eval_batch_size,
                                                       n_avg=n_avg,
                                                       return_all=True)
    return curr_model_activations, curr_model, curr_cfg


def load_cached_metric_scores(metric, layer, rnn_type, activation, mode, eval_arena_size=100, cv_type="elasticnet_max"):
    from mec_hpc_investigations.core.default_dirs import BASE_DIR_RESULTS, CAITLIN2D_MODEL_BORDERGRID_RESULTS
    curr_scores = \
    np.load(os.path.join(BASE_DIR_RESULTS, f"arena{eval_arena_size}_model_{metric}scores.npz"), allow_pickle=True)[
        'arr_0'][()]
    model_key = f"{rnn_type.lower()}_{activation}_{mode}"
    if (layer == "g") and (model_key in curr_scores.keys()):
        curr_scores = curr_scores[model_key]
    else:
        # TODO: extend to other settings if we use a different neural mapping strategy
        train_frac = 0.2
        num_train_test_splits = 10
        num_cv_splits = 2
        suffix = f"{cv_type}_caitlin2darena{eval_arena_size}_trainfrac{train_frac}"
        suffix += f"_numtrsp{num_train_test_splits}_numcvsp{num_cv_splits}"
        suffix += "_fixedinteranimal"
        curr_scores = np.load(os.path.join(CAITLIN2D_MODEL_BORDERGRID_RESULTS,
                                           f"{rnn_type}_bestneuralpredlayer_bordergridscores_{suffix}.npz"),
                              allow_pickle=True)['arr_0'][()][f"{rnn_type}_{activation}_{mode}"][layer][
            f"{metric}scores"]
    return curr_scores


def configure_options(run_ID: str,
                      save_dir=None,
                      dataset=None,
                      arena_size=None,
                      activation: str = "relu",
                      batch_size: int = 200,
                      bin_side_in_m: float = 0.05,
                      box_height_in_m: float = 2.2,
                      box_width_in_m: float = 2.2,
                      const_velocity_1d=False,
                      cue_2d_input_kwargs=None,
                      cue_input_mode_1d=None,
                      cue_input_only=False,
                      env_1d: bool=False,
                      initializer: str = 'glorot_uniform',
                      is_periodic: bool = False,
                      learning_rate: float = 1e-4,
                      Ng: int = 4096,
                      Np: int = 512,
                      n_epochs: int = 100,
                      n_recurrent_units_to_sample: int = 64,
                      n_grad_steps_per_epoch: int = 1000,
                      n_place_fields_per_cell: int = 1,
                      place_field_loss: str = 'crossentropy',
                      place_field_values: str = 'gaussian',
                      place_field_normalization: str = 'local',
                      place_cell_rf: float = 0.12,
                      pc_k=None,
                      pc_activation="relu",
                      pc_rnn_func=None,
                      pc_rnn_initial_state=False,
                      optimizer: str = 'adam',
                      seed: int = 0,
                      readout_dropout: float = 0.,
                      recurrent_dropout: float = 0.,
                      reward_zone_size=None,
                      reward_zone_prob=1.0,
                      # 15 cm from boundaries as in Kiah's paper
                      reward_zone_x_offset=0.15,
                      reward_zone_y_offset=0.15,
                      reward_zone_min_x=None,
                      reward_zone_max_x=None,
                      reward_zone_min_y=None,
                      reward_zone_max_y=None,
                      reward_zone_as_input=True,
                      reward_zone_navigate_timesteps=None,
                      rnn_type: str = "rnn",
                      sequence_length: int = 20,
                      surround_scale: float = 2.,
                      weight_decay: float = 1e-4,
                      ) -> Options:

    options = Options()

    if save_dir is None:
        options.save_dir = BASE_DIR_MODELS
    else:
        options.save_dir = save_dir

    if cue_2d_input_kwargs is not None:
        options.cue_2d_input_kwargs = cue_2d_input_kwargs
        options.cue_input_only = cue_input_only

    if reward_zone_size is not None:
        options.reward_zone_size = reward_zone_size
        assert (reward_zone_prob is not None)
        options.reward_zone_prob = reward_zone_prob
        assert (reward_zone_x_offset is not None)
        options.reward_zone_x_offset = reward_zone_x_offset
        assert (reward_zone_y_offset is not None)
        options.reward_zone_y_offset = reward_zone_y_offset
        options.reward_zone_min_x = reward_zone_min_x
        options.reward_zone_max_x = reward_zone_max_x
        options.reward_zone_min_y = reward_zone_min_y
        options.reward_zone_max_y = reward_zone_max_y
        options.reward_zone_as_input = reward_zone_as_input
        options.reward_zone_navigate_timesteps = reward_zone_navigate_timesteps

    options.place_field_loss = place_field_loss
    options.place_field_values = place_field_values
    options.place_field_normalization = place_field_normalization

    options.batch_size = batch_size  # number of trajectories per batch
    options.bin_side_in_m = bin_side_in_m
    options.initializer = initializer
    options.is_periodic = is_periodic
    options.learning_rate = learning_rate  # gradient descent learning rate
    options.n_epochs = n_epochs  # number of training epochs
    options.n_recurrent_units_to_sample = n_recurrent_units_to_sample
    options.n_grad_steps_per_epoch = n_grad_steps_per_epoch  # batches per epoch
    options.n_place_fields_per_cell = n_place_fields_per_cell

    options.Ng = Ng  # number of recurrent units
    options.Np = Np
    options.optimizer = optimizer
    if options.place_field_values == 'cartesian':
        assert options.Np == 2
    options.place_cell_rf = place_cell_rf  # width of place cell center tuning curve (m)
    options.rnn_type = rnn_type  # RNN or LSTM
    options.readout_dropout = readout_dropout
    options.recurrent_dropout = recurrent_dropout
    options.sequence_length = sequence_length  # number of steps in trajectory
    options.surround_scale = surround_scale  # if DoG, ratio of sigma2^2 to sigma1^2
    options.activation = activation  # recurrent nonlinearity
    options.weight_decay = weight_decay  # strength of weight decay on recurrent weights
    options.is_periodic = is_periodic  # trajectories with periodic boundary conditions
    options.seed = seed
    if (arena_size is None) and (not env_1d):
        options.box_width_in_m = box_width_in_m  # width of training environment (meters)
        options.box_height_in_m = box_height_in_m  # height of training environment (metrs)
    else:
        if dataset is None:
            if env_1d:
                # faster to load saved out packaged data
                # from mec.core.default_dirs import CAITLIN1D_VR_PACKAGED
                # print(f"Loading 1d environment parameters")
                # dataset = pickle.load(open(CAITLIN1D_VR_PACKAGED, "rb"))

                # we will load cached position bins to save memory during training
                dataset = None
            else:
                from mec_hpc_investigations.neural_data.datasets import CaitlinDatasetWithoutInertial, RewardDataset
                print(f"Loading 2d environment parameters for arena of size {arena_size} cm")
                if arena_size <= 100:
                    print(f"Loading Caitlin arena params")
                    dataset_obj = CaitlinDatasetWithoutInertial()
                    dataset_obj.package_data()
                    dataset = dataset_obj.packaged_data
                elif arena_size == 150:
                    print(f"Loading Butler Kiah arena params")
                    # the environment is the same between the free foraging and task datasets
                    dataset = RewardDataset(dataset="free_foraging")
                    dataset.package_data()
                else:
                    raise ValueError

        if env_1d:
            options.const_velocity_1d = const_velocity_1d
            options.cue_input_mode_1d = cue_input_mode_1d
            options.cue_input_only = cue_input_only
            options, pos_bins = set_cfg_from_vr1ddata(dataset=dataset,
                                                      cfg=options,
                                                      return_bins=True,
                                                      meters_convert=True)
            options.pos_bins = pos_bins
        else:
            assert (const_velocity_1d is False)
            assert (cue_input_mode_1d is None)
            options, arena_x_bins, arena_y_bins = set_cfg_from_data(dataset=dataset,
                                                                    cfg=options,
                                                                    arena_size=arena_size,
                                                                    return_bins=True,
                                                                    meters_convert=True)
            options.arena_x_bins = arena_x_bins
            options.arena_y_bins = arena_y_bins

    set_env_dims(options=options)

    options.run_ID = run_ID
    print(f"Configured these options {vars(options)}")
    return options


def configure_model(options: Options):

    place_cells = PlaceCells(options=options)
    rnn_type = options.rnn_type
    if rnn_type.lower() == "rnn":
        model = RNN(options=options, place_cells=place_cells)
    elif rnn_type.lower() == "lstm":
        model = LSTM(options=options, place_cells=place_cells)
    elif rnn_type.lower() == "ugrnn":
        model = UGRNN(options=options, place_cells=place_cells)
    elif rnn_type.lower() == "vanillarnn":
        model = VanillaRNN(options=options, place_cells=place_cells)
    elif rnn_type.lower() == "gru":
        model = GRU(options=options, place_cells=place_cells)
    elif rnn_type.lower() == "baninornn":
        model = BaninoRNN(options=options, place_cells=place_cells)
    elif rnn_type.lower() == "rewardrnn":
        model = RewardRNN(options=options, place_cells=place_cells)
    elif rnn_type.lower() == "rewardlstm":
        model = RewardLSTM(options=options, place_cells=place_cells)
    elif rnn_type.lower() == "rewardlstm2":
        model = RewardLSTM2(options=options, place_cells=place_cells)
    elif rnn_type.lower() == "rewardugrnn2":
        model = RewardUGRNN2(options=options, place_cells=place_cells)
    elif rnn_type.lower() == "lstmpcdense":
        assert (options.place_cell_predict is True)
        model = LSTMPCDense(options=options, place_cells=place_cells)
    elif rnn_type.lower() == "lstmpcrnn":
        assert (options.place_cell_predict is True)
        model = LSTMPCRNN(options=options, place_cells=place_cells)
    # elif rnn_type.lower() == "ugrnnpcdense":
    #     assert(options.place_cell_predict is True)
    #     model = UGRNNPCDense(options=options, place_cells=place_cells)
    else:
        raise ValueError
    return model


def load_trained_model(options=None,
                       save_dir=None,
                       dataset=None,
                       env_1d=False,
                       const_velocity_1d=False,
                       cue_input_mode_1d=None,
                       rnn_type="rnn",
                       activation="relu",
                       nmf_components=9,
                       arena_size=None,
                       random_init=False,
                       place_cell_identity=False,
                       place_cell_predict=False,
                       num_pc_pred=512,
                       pc_k=None,
                       pc_activation="relu",
                       pc_rnn_func=None,
                       pc_rnn_initial_state=False,
                       n_epochs=100,
                       n_steps=1000,
                       place_cell_rf=0.12,
                       banino_place_cell=False,
                       Np=512,
                       Ng=4096,
                       Nhdc=None,
                       hdc_concentration=20,
                       weight_decay=1e-4,
                       learning_rate=1e-4,
                       batch_size=200,
                       sequence_length=20,
                       optimizer_class="adam",
                       banino_rnn_type="lstm",
                       banino_dropout_rate=0.5,
                       banino_rnn_nunits=128,
                       clipvalue=None,
                       cue_2d_input_kwargs=None,
                       cue_input_only=False,
                       ckpt_file=None,
                       run_ID=None,
                       **reward_zone_kwargs):
    if options is None:
        options = configure_options(save_dir=save_dir,
                                    dataset=dataset,
                                    rnn_type=rnn_type,
                                    activation=activation,
                                    arena_size=arena_size,
                                    env_1d=env_1d,
                                    const_velocity_1d=const_velocity_1d,
                                    cue_input_mode_1d=cue_input_mode_1d,
                                    place_cell_identity=place_cell_identity,
                                    place_cell_predict=place_cell_predict,
                                    num_pc_pred=num_pc_pred,
                                    pc_k=pc_k,
                                    pc_activation=pc_activation,
                                    pc_rnn_func=pc_rnn_func,
                                    pc_rnn_initial_state=pc_rnn_initial_state,
                                    cue_2d_input_kwargs=cue_2d_input_kwargs,
                                    cue_input_only=cue_input_only,
                                    n_epochs=n_epochs,
                                    n_steps=n_steps,
                                    place_cell_rf=place_cell_rf,
                                    banino_place_cell=banino_place_cell,
                                    Np=Np,
                                    Ng=Ng,
                                    Nhdc=Nhdc,
                                    hdc_concentration=hdc_concentration,
                                    weight_decay=weight_decay,
                                    learning_rate=learning_rate,
                                    batch_size=batch_size,
                                    sequence_length=sequence_length,
                                    optimizer_class=optimizer_class,
                                    banino_rnn_type=banino_rnn_type,
                                    banino_dropout_rate=banino_dropout_rate,
                                    banino_rnn_nunits=banino_rnn_nunits,
                                    clipvalue=clipvalue,
                                    run_ID=run_ID,
                                    **reward_zone_kwargs)
    else:
        print(vars(options))

    if rnn_type.lower() != "nmf":
        trained_model = configure_model(options, rnn_type=rnn_type)
        if not random_init:
            ckpt_dir = os.path.join(options.save_dir, options.run_ID, "ckpts")
            # TODO: come up with better solution here
            if ("pc_rnn_func" not in ckpt_dir) and ("BaninoRNN" not in ckpt_dir):
                ckpt_dir = ckpt_dir.replace("rnn", "RNN")
            assert (os.path.isdir(ckpt_dir) is True)
            ckpt = tf.train.Checkpoint(step=tf.Variable(0),
                                       optimizer=tf.keras.optimizers.Adam(learning_rate=options.learning_rate),
                                       net=trained_model)
            ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=500)
            if ckpt_file is None:
                latest_ckpt_path = ckpt_manager.latest_checkpoint
            else:
                latest_ckpt_path = os.path.join(ckpt_dir, ckpt_file)
            print(f"Loading ckpt from {latest_ckpt_path}")
            ckpt.restore(latest_ckpt_path).assert_existing_objects_matched()
    else:
        # train NMF
        assert (random_init is False)
        from tqdm import tqdm
        print("Training NMF")
        from sklearn.decomposition import NMF
        nmf = NMF(n_components=nmf_components, init='random', random_state=0)
        place_cells = PlaceCells(options)
        trajectory_generator = TrajectoryGenerator(options, place_cells, trajectory_seed=None)
        gen = trajectory_generator.get_generator()
        n_epochs = 1
        n_steps = 1000
        n_avg = n_epochs * n_steps
        p = np.zeros([n_avg, options.batch_size * options.sequence_length, options.Np])
        pos = np.zeros([n_avg, options.batch_size * options.sequence_length, 2])
        for index in tqdm(range(n_avg)):
            inputs, pc_outputs, pos_batch = next(gen)
            p_batch = np.reshape(pc_outputs, (-1, options.Np))
            pos_batch = np.reshape(pos_batch, (-1, 2))
            p[index] = p_batch
            pos[index] = pos_batch
        p = p.reshape((-1, options.Np))
        pos = pos.reshape((-1, 2))
        if env_1d:
            p_activations = scipy.stats.binned_statistic(x=pos[:, 0],
                                                         values=p.T,
                                                         statistic='mean',
                                                         bins=options.pos_bins)[0]
            # put cells in last dimension
            p_2d = np.transpose(p_activations, (1, 0))
        else:
            if not (hasattr(options, 'arena_x_bins') and hasattr(options, 'arena_y_bins')):
                assert (arena_size is None)  # training on big 2.2 x 2.2 m arena
                assert (options.box_width_in_m == options.box_height_in_m)
                assert (options.box_width_in_m == 2.2)
                # 5 cm bins
                nbins = (int)((options.box_width_in_m * 100) / 5.0)
                print(f"Binning into {nbins} bins for training data")
                options.arena_x_bins = np.linspace(options.min_x, options.max_x, nbins + 1, endpoint=True)
                options.arena_y_bins = np.linspace(options.min_y, options.max_y, nbins + 1, endpoint=True)

            p_activations = scipy.stats.binned_statistic_2d(x=pos[:, 0],
                                                            y=pos[:, 1],
                                                            values=p.T,
                                                            statistic='mean',
                                                            bins=[options.arena_x_bins, options.arena_y_bins])[0]
            # put cells in last dimension
            p_activations = np.transpose(p_activations, (1, 2, 0))
            p_2d = p_activations.reshape((-1, options.Np))

        trained_model = nmf.fit(p_2d)
    return trained_model
