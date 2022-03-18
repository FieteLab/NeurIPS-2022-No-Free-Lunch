import numpy as np
import os, copy
from mec_hpc_investigations.core.utils import all_disjoint, get_shape_2d
from joblib import delayed, Parallel
from tqdm import tqdm

def generate_train_test_splits(num_states,
                               split_start_seed=0,
                               num_splits=10,
                               train_frac=0.5,
                               val_frac=None,
                               shape_2d=None):
    if isinstance(train_frac, str):
        assert(val_frac is None) # very structured splits so no val_frac, good for testing after the fact
        num_rows, num_cols = get_shape_2d(num_states=num_states,
                                          shape_2d=shape_2d)

        idx_mat = np.arange(num_states).reshape((num_rows, num_cols))
        train_test_splits = []
        if train_frac == "topbottom":
            half_1 = idx_mat[:(num_rows//2), :].flatten()
            half_2 = idx_mat[(num_rows//2):, :].flatten()
        elif train_frac == "leftright":
            half_1 = idx_mat[:, :(num_cols//2)].flatten()
            half_2 = idx_mat[:, (num_cols//2):].flatten()
        elif train_frac == "diag1":
            half_1 = idx_mat[np.triu_indices(n=num_rows,k=0,m=num_cols)].flatten()
            half_2 = idx_mat[np.tril_indices(n=num_rows,k=-1,m=num_cols)].flatten()
        elif train_frac == "diag2":
            idx_mat_2 = np.fliplr(idx_mat)
            half_1 = idx_mat_2[np.triu_indices(n=num_rows,k=0,m=num_cols)].flatten()
            half_2 = idx_mat_2[np.tril_indices(n=num_rows,k=-1,m=num_cols)].flatten()
        elif train_frac == "quad":
            quad1_test = idx_mat[:(num_rows//2), :(num_cols//2)].flatten()
            quad1_train = np.array([x for x in idx_mat.flatten() if x not in quad1_test])
            # check that it is a complete set
            assert(set(list(quad1_train)+list(quad1_test)) == set(np.arange(num_states)))
            quad2_test = idx_mat[:(num_rows//2), (num_cols//2):].flatten()
            quad2_train = np.array([x for x in idx_mat.flatten() if x not in quad2_test])
            # check that it is a complete set
            assert(set(list(quad2_train)+list(quad2_test)) == set(np.arange(num_states)))
            quad3_test = idx_mat[(num_rows//2):, :(num_cols//2)].flatten()
            quad3_train = np.array([x for x in idx_mat.flatten() if x not in quad3_test])
            # check that it is a complete set
            assert(set(list(quad3_train)+list(quad3_test)) == set(np.arange(num_states)))
            quad4_test = idx_mat[(num_rows//2):, (num_cols//2):].flatten()
            quad4_train = np.array([x for x in idx_mat.flatten() if x not in quad4_test])
            # check that it is a complete set
            assert(set(list(quad4_train)+list(quad4_test)) == set(np.arange(num_states)))
            # check that all quadrants are pairwise distinct
            assert(all_disjoint([quad1_test, quad2_test, quad3_test, quad4_test]) is True)
            train_test_splits.append({'train': quad1_train, 'test': quad1_test})
            train_test_splits.append({'train': quad2_train, 'test': quad2_test})
            train_test_splits.append({'train': quad3_train, 'test': quad3_test})
            train_test_splits.append({'train': quad4_train, 'test': quad4_test})
        else:
            raise ValueError

        if train_frac != "quad":
            # check that it is a complete set
            assert(set(list(half_1)+list(half_2)) == set(np.arange(num_states)))
            train_test_splits.append({'train': half_1, 'test': half_2})
            train_test_splits.append({'train': half_2, 'test': half_1})

    elif train_frac > 0:
        assert(train_frac <= 1)
        train_test_splits = []
        for s in range(num_splits):
            rand_idx = np.random.RandomState(seed=(split_start_seed + s)).permutation(num_states)
            num_train = (int)(np.ceil(train_frac*len(rand_idx)))
            train_idx = rand_idx[:num_train]
            if val_frac is not None:
                assert(val_frac > 0)
                assert(val_frac <= 1)
                num_val = (int)(np.ceil(val_frac*len(rand_idx)))
                assert((num_train + num_val) <= len(rand_idx))
                val_idx = rand_idx[num_train:(num_train+num_val)]
                test_idx = rand_idx[(num_train+num_val):]
                curr_sp = {'train': train_idx,
                           'val': val_idx,
                           'test': test_idx}
            else:
                test_idx = rand_idx[num_train:]
                curr_sp = {'train': train_idx,
                           'test': test_idx}
            train_test_splits.append(curr_sp)
    else:
        print("Train fraction is 0, make sure your map has no parameters!")
        # we apply no random permutation in this case as there is no training of parameters
        # (e.g. rsa)
        assert(val_frac is None) # no need for val frac here since no parameters to cv
        train_test_splits = [{'train': np.array([], dtype=int), 'test': np.arange(num_states)}]
    return train_test_splits


def nan_filter(source_resp, target_resp):
    """Helper function that first filters stimuli across all cells within a given animal
    so that it is all non-Nan. Then, filters the non-NaN stimuli across both source and target response."""
    # stimuli across all cells in a given animal/model that are non-Nan
#     source_include = ~np.isnan(source_resp).any(axis=-1)
#     target_include = ~np.isnan(target_resp).any(axis=-1)
    source_include = np.isfinite(source_resp).all(axis=-1)
    target_include = np.isfinite(target_resp).all(axis=-1)
    # compare across stimuli where both are non-NaN
    both_include = np.logical_and(source_include, target_include)
    source_resp = source_resp[both_include]
    target_resp = target_resp[both_include]
    return source_resp, target_resp

def prep_data_2d(X, Y):
    """Helper function that ensures data is non-NaN and flattened to be (num_stimuli, num_units)."""
    X = X.reshape((-1, X.shape[-1]))
    Y = Y.reshape((-1, Y.shape[-1]))
    X, Y = nan_filter(X, Y)
    assert X.shape[0] == Y.shape[0]
    return X, Y

def return_mean_sp_scores(map_type,
                          map_kwargs,
                          train_test_sp,
                          X,
                          Y,
                          first_X=None,
                          return_responses=False,
                          scorer=None,
                          apply_sac_mask=False,
                          shape_2d=None):
    from mec_hpc_investigations.neural_mappers.pipeline_neural_map import PipelineNeuralMap
    """Returns the scores per train/test split for each metric type."""
    mean_scores = {"corr": []}
    if scorer is not None:
        mean_scores["shuffle_corr"] = []
    if return_responses:
        mean_scores["resp"] = []
    if isinstance(map_kwargs, list):
        # a different kwarg per train test split
        assert(len(map_kwargs) == len(train_test_sp))
    for curr_sp_idx, curr_sp in enumerate(train_test_sp):
        train_idx = curr_sp["train"]
        test_idx = curr_sp["test"]

        curr_map_kwargs = map_kwargs[curr_sp_idx] if isinstance(map_kwargs, list) else copy.deepcopy(map_kwargs)
        curr_map = PipelineNeuralMap(map_type=map_type,
                                     map_kwargs=curr_map_kwargs)
        curr_map.fit(X=X[train_idx], Y=Y[train_idx], first_X=first_X[train_idx] if first_X is not None else None)
        if first_X is not None:
            Y_pred = curr_map.predict(first_X[test_idx])
        else:
            Y_pred = curr_map.predict(X[test_idx])

        if scorer is None:
            curr_sp_score = curr_map.score(Y=Y[test_idx],
                                           Y_pred=Y_pred)
            mean_scores["corr"].append(curr_sp_score)
        else:
            from mec_hpc_investigations.neural_fits.comparisons import _get_sac
            # we compute the autocorrelation from the rate maps on all positions and then evaluate on all positions
            # note the mapping was trained on a subset of position bins
            if first_X is not None:
                Y_pred_all = curr_map.predict(first_X)
            else:
                Y_pred_all = curr_map.predict(X)

            assert(Y.shape == Y_pred_all.shape)
            assert(len(Y_pred_all.shape) == 2)
            num_pred_cells = Y_pred_all.shape[1]
            num_pred_states = Y_pred_all.shape[0]
            num_rows, num_cols = get_shape_2d(num_states=num_pred_states,
                                              shape_2d=shape_2d)
            Y_sac = _get_sac(scorer=scorer,
                             resp=Y.reshape((num_rows, num_cols, num_pred_cells)),
                             apply_sac_mask=apply_sac_mask)
            Y_pred_sac = _get_sac(scorer=scorer,
                                  resp=Y_pred_all.reshape((num_rows, num_cols, num_pred_cells)),
                                  apply_sac_mask=apply_sac_mask)
            curr_sp_score = curr_map.score(Y=Y_sac.reshape((-1, num_pred_cells)),
                                           Y_pred=Y_pred_sac.reshape((-1, num_pred_cells)))

            mean_scores["corr"].append(curr_sp_score)
            Y_pred_sh = copy.deepcopy(Y_pred_all)
            # shuffle along the bins axis for just the test set bins, across ALL cells
            shuffle_s = curr_sp_idx + len(train_test_sp)
            Y_pred_shuffle_train = Y_pred_sh[train_idx]
            Y_pred_shuffle_test = Y_pred_sh[test_idx]
            np.random.RandomState(seed=shuffle_s).shuffle(Y_pred_shuffle_test)
            Y_pred_shuffle = np.zeros_like(Y_pred_all) + np.NaN
            Y_pred_shuffle[train_idx] = Y_pred_shuffle_train
            Y_pred_shuffle[test_idx] = Y_pred_shuffle_test
            assert(not np.array_equal(Y_pred_shuffle, Y_pred_all))
            assert(Y_pred_shuffle.shape == Y_pred_all.shape)
            # then compute autocorrelation
            Y_pred_shuffle_sac = _get_sac(scorer=scorer,
                                          resp=Y_pred_shuffle.reshape((num_rows, num_cols, num_pred_cells)),
                                          apply_sac_mask=apply_sac_mask)

            assert(not np.array_equal(Y_pred_shuffle_sac, Y_pred_sac))
            curr_sp_shuffle_score = curr_map.score(Y=Y_sac.reshape((-1, num_pred_cells)),
                                                   Y_pred=Y_pred_shuffle_sac.reshape((-1, num_pred_cells)))

            mean_scores["shuffle_corr"].append(curr_sp_shuffle_score)

        if return_responses:
            if first_X is not None:
                Y_pred_all = curr_map.predict(first_X)
            else:
                Y_pred_all = curr_map.predict(X)
            # we save on all positions to make visualizing easy when reshaping back to 2D per cell
            store_dict = {"Y_pred": Y_pred_all, "Y": Y, "train_idx": train_idx, "test_idx": test_idx}
            if scorer is not None:
                store_dict["Y_sac"] = Y_sac
                store_dict["Y_pred_sac"] = Y_pred_sac
                store_dict["Y_pred_shuffle"] = Y_pred_shuffle
                store_dict["Y_pred_shuffle_sac"] = Y_pred_shuffle_sac
            mean_scores["resp"].append(store_dict)

    # average across train/test splits for each metric
    for metric_type in mean_scores.keys():
        if metric_type != "resp":
            mean_scores[metric_type] = np.nanmean(np.stack(mean_scores[metric_type], axis=0), axis=0)

    return mean_scores

def package_scores(scores, cell_ids):
    """If we have scores per unit (e.g. NOT RSA), then we will associate
    each score with a unit as a single xarray."""
    if len(scores.shape) == 1:
        import xarray as xr
        scores = xr.DataArray(scores,
                             coords={"units": cell_ids},
                             dims=["units"])
    return scores

def get_val_scores(m_sel,
                  curr_sp_idx,
                  num_train_test_splits=10,
                  score_key="val_scores"):
    curr_sp_val_scores = []
    for m_e in m_sel:
        if isinstance(m_e[score_key], float):
            assert(np.isnan(m_e[score_key]))
            curr_sp_val_scores.append(m_e[score_key])
        else:
            assert(isinstance(m_e[score_key], list))
            assert(len(m_e[score_key]) == num_train_test_splits)
            curr_sp_val_scores.append(m_e[score_key][curr_sp_idx])
    curr_sp_val_scores = np.array(curr_sp_val_scores)
    assert(len(curr_sp_val_scores) == len(m_sel))
    return curr_sp_val_scores

def find_parameters_max(curr_sp_val_scores, m_sel):
    curr_sp_hs_idx = np.nanargmax(curr_sp_val_scores)
    m_sel_curr_sp = m_sel[curr_sp_hs_idx]
    alpha_sel = m_sel_curr_sp["alpha"]
    l1_ratio_sel = m_sel_curr_sp["l1_ratio"]
    alpha_renorm_sel = m_sel_curr_sp["alpha_renorm"][0]
    return m_sel_curr_sp, alpha_sel, l1_ratio_sel, alpha_renorm_sel

def make_dict(parallel_results, all_cells):
    cc_map_kwargs_per_cell = {}
    for idx, cell_id in enumerate(all_cells):
        curr_cc_results = parallel_results[idx]
        assert(curr_cc_results["cell_id"] == cell_id)
        cc_map_kwargs_per_cell[cell_id] = curr_cc_results
    return cc_map_kwargs_per_cell

def gen_max_worker(
    cell_id,
    dataset_name="caitlin2dwithoutinertial",
    l1_ratio_val=None,
    train_frac_range=[0.2],
    num_train_test_splits=10,
    num_cv_splits=2,
):
    from mec_hpc_investigations.neural_fits.sample_elasticnet_per_neuron import construct_filename as ld_filename
    worker_results = {"cell_id": cell_id, "map_kwargs": [], "test_scores": []}
    if dataset_name == "caitlin2dwithoutinertial":
        from mec_hpc_investigations.core.default_dirs import CAITLIN2D_INTERANIMAL_CC_MAP
        load_dir = CAITLIN2D_INTERANIMAL_CC_MAP
        arena_size = 100
    elif dataset_name.startswith("caitlinhpc"):
        from mec_hpc_investigations.core.default_dirs import CAITLINHPC_INTERANIMAL_SAMPLE_RESULTS
        load_dir = CAITLINHPC_INTERANIMAL_SAMPLE_RESULTS
        if dataset_name == "caitlinhpc62":
            arena_size = 62
        elif dataset_name == "caitlinhpc50":
            arena_size = 50
        else:
            raise ValueError
    else:
        assert(dataset_name in ["ofreward_combined", "of_only", "reward_only"])
        from mec_hpc_investigations.core.default_dirs import OFREWARD_COMBINED_INTERANIMAL_SAMPLE_RESULTS
        load_dir = OFREWARD_COMBINED_INTERANIMAL_SAMPLE_RESULTS
        arena_size = 150

    curr_fn = ld_filename(
        target_cell_id=cell_id,
        dataset_name=dataset_name,
        arena_size=arena_size,
        train_frac_range=train_frac_range,
        num_cv_splits=num_cv_splits
    )
    m_sel = np.load(
        os.path.join(load_dir, curr_fn), allow_pickle=True
    )["arr_0"][()]
    if l1_ratio_val is not None:
        curr_l1_ratio = np.array([m_e["l1_ratio"] for m_e in m_sel])
        curr_idx = np.where(curr_l1_ratio == l1_ratio_val)[0]
        assert(len(curr_idx) == 99)
        m_sel = m_sel[curr_idx]

    for curr_sp_idx in range(num_train_test_splits):
        curr_sp_val_scores = get_val_scores(
            m_sel=m_sel,
            curr_sp_idx=curr_sp_idx,
            num_train_test_splits=num_train_test_splits,
        )
        # for the current train/test split:
        # find the parameters that are the maximum on val set
        m_sel_curr_sp, alpha_sel, l1_ratio_sel, alpha_renorm_sel = find_parameters_max(
            curr_sp_val_scores=curr_sp_val_scores, m_sel=m_sel
        )
        if l1_ratio_val is not None:
            assert(l1_ratio_sel == l1_ratio_val)
        curr_sp_map_kwargs = {
            "regression_type": "ElasticNet",
            "regression_kwargs": {"alpha": alpha_renorm_sel, "l1_ratio": l1_ratio_sel},
        }
        worker_results["map_kwargs"].append(curr_sp_map_kwargs)
        test_scores_sel = m_sel_curr_sp["test_scores"][curr_sp_idx]
        worker_results["test_scores"].append(test_scores_sel)
    return worker_results

def construct_map_kwargs_per_cell(dataset_name="caitlin2dwithoutinertial",
                                  n_jobs=20, **kwargs):
    if dataset_name == "caitlin2dwithoutinertial":
        from mec_hpc_investigations.core.default_dirs import CAITLIN_BASE_DIR
        all_cells = np.load(os.path.join(CAITLIN_BASE_DIR, "cell_ids_2d.npz"), allow_pickle=True)["arr_0"][()]
    elif dataset_name.startswith("caitlinhpc"):
        from mec_hpc_investigations.core.default_dirs import CAITLIN2D_HPC
        d = np.load(os.path.join(CAITLIN2D_HPC, "dataset.npz"), allow_pickle=True)["arr_0"][()]
        if dataset_name == "caitlinhpc62":
            arena_size = 62
        elif dataset_name == "caitlinhpc50":
            arena_size = 50
        else:
            raise ValueError
        curr_animals = list(d[arena_size].keys())
        all_cells = []
        for a in curr_animals:
            for c in d[arena_size][a]['cell_ids']:
                all_cells.append(f"{a}_{c}")
        all_cells = np.array(all_cells)
    else:
        assert(dataset_name in ["ofreward_combined", "of_only", "reward_only"])
        from mec_hpc_investigations.core.default_dirs import REWARD_BASE_DIR
        all_cells = np.load(os.path.join(REWARD_BASE_DIR, "cell_ids_2d.npz"), allow_pickle=True)["arr_0"][()]
    parallel_results = Parallel(n_jobs=n_jobs)(delayed(gen_max_worker)(cell_id=cell_id, dataset_name=dataset_name, **kwargs) for cell_id in tqdm(all_cells))
    map_kwargs_per_cell = make_dict(parallel_results=parallel_results, all_cells=all_cells)
    return map_kwargs_per_cell

def get_max_layer_fits(results_dict_alllayers, eval_arena_size):
    from mec_hpc_investigations.neural_data.utils import unit_concat
    scores_arr = []
    layers_arr = []
    for curr_layer, results_dict in results_dict_alllayers.items():
        curr_score_raw = unit_concat(results_dict, arena_size=eval_arena_size, inner_key="corr")
        curr_score = curr_score_raw[np.isfinite(curr_score_raw)]
        scores_arr.append(np.median(curr_score))
        layers_arr.append(curr_layer)
    layer_max = layers_arr[np.nanargmax(scores_arr)]
    results_dict = results_dict_alllayers[layer_max]
    score_raw = unit_concat(results_dict, arena_size=eval_arena_size, inner_key="corr")
    curr_neural_fits = score_raw[np.isfinite(score_raw)]
    return curr_neural_fits, layer_max
