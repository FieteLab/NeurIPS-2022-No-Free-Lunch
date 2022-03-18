import numpy as np
import os, copy
import itertools
from mec_hpc_investigations.neural_fits.utils import generate_train_test_splits, prep_data_2d, return_mean_sp_scores, package_scores
from mec_hpc_investigations.models.utils import get_model_activations

def fit_subroutine(X, Y,
             map_type, map_kwargs,
             train_frac, num_train_test_splits,
             split_start_seed=0,
             first_X=None,
             return_responses=False,
             scorer=None, apply_sac_mask=False,
             shape_2d=None):

    if first_X is not None:
        first_X, X = prep_data_2d(X=first_X, Y=X)
        X, Y = prep_data_2d(X=X, Y=Y)
    else:
        X, Y = prep_data_2d(X=X, Y=Y)

    # we generate train/test splits for each source, target pair
    # since the stimuli can be different for each pair (when we don't smooth firing rates)
    train_test_sp = generate_train_test_splits(num_states=X.shape[0],
                                               train_frac=train_frac,
                                               num_splits=num_train_test_splits,
                                               shape_2d=shape_2d,
                                               split_start_seed=split_start_seed
                                              )

    if map_type in ["linearshape", "cca", "proc", "cka"]:
        from netrep.metrics import LinearMetric, LinearCKA
        assert(return_responses is False)
        assert(first_X is None)
        assert(scorer is None)

        mean_scores = {"dist": []}
        for curr_sp_idx, curr_sp in enumerate(train_test_sp):
            train_idx = curr_sp["train"]
            test_idx = curr_sp["test"]

            if map_type == "linearshape":
                assert("alpha" in map_kwargs.keys())
                curr_metric = LinearMetric(**map_kwargs)
            elif map_type == "proc":
                assert("alpha" not in map_kwargs.keys())
                # Rotationally invariant metric (fully regularized).
                curr_metric = LinearMetric(alpha=1.0, **map_kwargs)
            elif map_type == "cca":
                assert("alpha" not in map_kwargs.keys())
                # Linearly invariant metric (no regularization).
                curr_metric = LinearMetric(alpha=0.0, **map_kwargs)
            else:
                assert(map_type == "cka")
                curr_metric = LinearCKA(**map_kwargs)

            curr_metric.fit(X[train_idx], Y[train_idx])
            dist = curr_metric.score(X[test_idx], Y[test_idx])
            # unlike the other metrics, we do not average over train test splits since this is our only source of variance
            mean_scores["dist"].append(dist)
        mean_scores["dist"] = np.array(mean_scores["dist"])
    else:
        mean_scores = return_mean_sp_scores(map_type=map_type,
                                              map_kwargs=map_kwargs,
                                              train_test_sp=train_test_sp,
                                              X=X,
                                              Y=Y,
                                              first_X=first_X,
                                              return_responses=return_responses,
                                              scorer=scorer,
                                              apply_sac_mask=apply_sac_mask,
                                              shape_2d=shape_2d)
    return mean_scores

def compare_animals(arena_sizes, spec_resp_agg,
                  dataset=None,
                  source_spec_resp_agg=None,
                  map_type="identity", map_kwargs={},
                  train_frac=0.0, num_train_test_splits=10, split_start_seed=0,
                  mode="pairwise", place_cell_resp=None,
                  weights_save_dir=None, return_responses=False,
                  scorer=None, apply_sac_mask=False,
                  shape_2d=None,
                  source_cell_ids=None,
                  target_cell_ids=None,
                  map_kwargs_per_cell=None):

    """Compare animals to each other with a given map"""
    if source_spec_resp_agg is None:
        source_spec_resp_agg = copy.deepcopy(spec_resp_agg)

    print("Comparing across animals")
    score_results = {}
    for arena_size in arena_sizes:

        score_results[arena_size] = {}
        arena_animals = list(spec_resp_agg[arena_size].keys())
        if len(arena_animals) > 1:
            for animal in arena_animals:
                score_results[arena_size][animal] = {}
                if mode == "pairwise":
                    assert(map_kwargs_per_cell is None) # currently unsupported since these kwargs were found via cross val in holdout mode
                    for animal_pair in itertools.permutations(arena_animals, r=2):
                        source_animal = animal_pair[0]
                        target_animal = animal_pair[1]
                        if target_animal == animal:
                            assert(weights_save_dir is None)
                            source_animal_resp = source_spec_resp_agg[arena_size][source_animal]["resp"]
                            if source_cell_ids is not None:
                                curr_source_cell_ids = source_spec_resp_agg[arena_size][source_animal]["cell_ids"]
                                # find the chosen cells that are present in the current source animal
                                sel_source_cells = list(set(curr_source_cell_ids) & set(source_cell_ids))
                                if len(sel_source_cells) == 0:
                                    # nothing to fit
                                    continue
                                sel_source_idxs = np.where(np.in1d(curr_source_cell_ids, sel_source_cells))[0]
                                source_animal_resp = source_animal_resp[:, :, sel_source_idxs]

                            target_animal_resp = spec_resp_agg[arena_size][target_animal]["resp"]
                            if target_cell_ids is not None:
                                curr_target_cell_ids = spec_resp_agg[arena_size][target_animal]["cell_ids"]
                                # find the chosen cells that are present in the current target animal
                                sel_target_cells = list(set(curr_target_cell_ids) & set(target_cell_ids))
                                if len(sel_target_cells) == 0:
                                    # nothing to fit
                                    continue
                                sel_target_idxs = np.where(np.in1d(curr_target_cell_ids, sel_target_cells))[0]
                                target_animal_resp = target_animal_resp[:, :, sel_target_idxs]

                            mean_scores = fit_subroutine(X=source_animal_resp, Y=target_animal_resp,
                                                       map_type=map_type, map_kwargs=map_kwargs,
                                                       train_frac=train_frac,
                                                       num_train_test_splits=num_train_test_splits,
                                                       split_start_seed=split_start_seed,
                                                       first_X=place_cell_resp,
                                                       return_responses=return_responses,
                                                       scorer=scorer, apply_sac_mask=apply_sac_mask,
                                                       shape_2d=shape_2d)

                            for metric_type in mean_scores.keys():
                                if metric_type not in score_results[arena_size][animal].keys():
                                    score_results[arena_size][animal][metric_type] = [mean_scores[metric_type]]
                                else:
                                    score_results[arena_size][animal][metric_type].append(mean_scores[metric_type])

                elif mode == "holdout":
                    """Concatenate neurons from source animal and fit to target."""
                    curr_source_animals = list(set(arena_animals) - set([animal]))
                    assert(animal not in curr_source_animals)
                    assert(len(curr_source_animals) == len(arena_animals) - 1)

                    mega_source_animal_resp = np.concatenate([source_spec_resp_agg[arena_size][source_animal]["resp"] for source_animal in curr_source_animals], axis=-1)
                    if source_cell_ids is not None:
                        curr_mega_source_cell_ids = np.concatenate([source_spec_resp_agg[arena_size][source_animal]["cell_ids"] for source_animal in curr_source_animals], axis=-1)
                        # find the chosen cells that are present in the mega source animal
                        sel_mega_source_cells = list(set(curr_mega_source_cell_ids) & set(source_cell_ids))
                        if len(sel_mega_source_cells) == 0:
                            # nothing to fit
                            continue
                        sel_mega_source_idxs = np.where(np.in1d(curr_mega_source_cell_ids, sel_mega_source_cells))[0]
                        mega_source_animal_resp = mega_source_animal_resp[:, :, sel_mega_source_idxs]

                    target_animal_resp = spec_resp_agg[arena_size][animal]["resp"]
                    if target_cell_ids is not None:
                        curr_target_cell_ids = spec_resp_agg[arena_size][animal]["cell_ids"]
                        # find the chosen cells that are present in the current target animal
                        sel_target_cells = list(set(curr_target_cell_ids) & set(target_cell_ids))
                        if len(sel_target_cells) == 0:
                            # nothing to fit
                            continue
                        sel_target_idxs = np.where(np.in1d(curr_target_cell_ids, sel_target_cells))[0]
                        target_animal_resp = target_animal_resp[:, :, sel_target_idxs]

                    if weights_save_dir is not None:
                        assert(num_train_test_splits==1)
                        assert(source_cell_ids is None) # since curr_source_animals is not the accurate list of source animals used if it is not None

                    subroutine_kwargs = {"X": mega_source_animal_resp, "map_type": map_type,
                                         "num_train_test_splits": num_train_test_splits,
                                         "split_start_seed": split_start_seed,
                                         "first_X": place_cell_resp,
                                         "return_responses": return_responses,
                                         "scorer": scorer, "apply_sac_mask": apply_sac_mask,
                                         "shape_2d": shape_2d}
                    if map_kwargs_per_cell is not None:
                        # no subselection allowed in this case, since these kwargs were found via cross validation on holdout animal between source and target
                        assert(source_cell_ids is None)
                        assert(target_cell_ids is None)
                        target_animal_cell_ids = spec_resp_agg[arena_size][animal]["cell_ids"]
                        mean_scores = {}
                        for n, curr_target_cell_id in enumerate(target_animal_cell_ids):
                            if weights_save_dir is not None:
                                map_kwargs_per_cell[curr_target_cell_id]["map_kwargs"]['weights_save_nm'] = os.path.join(weights_save_dir, f"weights_to_target_{curr_target_cell_id}.npz")
                            curr_subroutine_kwargs = copy.deepcopy(subroutine_kwargs)
                            if len(target_animal_resp.shape) == 3:
                                curr_subroutine_kwargs["Y"] = np.expand_dims(target_animal_resp[:, :, n], axis=-1)
                            elif len(target_animal_resp.shape) == 2:
                                curr_subroutine_kwargs["Y"] = np.expand_dims(target_animal_resp[:, n], axis=-1)
                            else:
                                raise ValueError
                            curr_subroutine_kwargs["map_kwargs"] = map_kwargs_per_cell[curr_target_cell_id]["map_kwargs"]
                            # can specify a train frac per cell, otherwise it globally uses the one passed in above
                            curr_subroutine_kwargs["train_frac"] = map_kwargs_per_cell[curr_target_cell_id].get("train_frac", train_frac)
                            curr_mean_scores = fit_subroutine(**curr_subroutine_kwargs)
                            mean_scores[curr_target_cell_id] = curr_mean_scores
                    else:
                        if weights_save_dir is not None:
                            map_kwargs['weights_save_nm'] = os.path.join(weights_save_dir, f"weights_to_target_{animal}.npz")
                        subroutine_kwargs["Y"] = target_animal_resp
                        subroutine_kwargs["map_kwargs"] = map_kwargs
                        subroutine_kwargs["train_frac"] = train_frac
                        mean_scores = fit_subroutine(**subroutine_kwargs)

                    if weights_save_dir is not None:
                        score_results[arena_size][animal] = {"scores": mean_scores, "sources": np.array([source_animal for source_animal in curr_source_animals])}
                    else:
                        score_results[arena_size][animal] = mean_scores
                else:
                    raise ValueError

        for animal in score_results[arena_size].keys():
            metrics = ["corr"] if map_kwargs_per_cell is not None else score_results[arena_size][animal].keys()
            for metric_type in metrics:
                if mode == "pairwise":
                    if metric_type != "resp":
                        # average across source animals
                        score_results[arena_size][animal][metric_type] = np.nanmean(np.stack(score_results[arena_size][animal][metric_type], axis=0), axis=0)

                if weights_save_dir is None:
                    if metric_type not in ["resp", "dist"]:
                        pkg_cell_ids = spec_resp_agg[arena_size][animal]["cell_ids"]
                        if target_cell_ids is not None:
                            # find the chosen cells that are present in the current target animal
                            sel_target_cells = list(set(pkg_cell_ids) & set(target_cell_ids))
                            # sanity check: since we are in this loop that means this animal had target units in it
                            assert(len(sel_target_cells) != 0)
                            sel_target_idxs = np.where(np.in1d(pkg_cell_ids, sel_target_cells))[0]
                            pkg_cell_ids = pkg_cell_ids[sel_target_idxs]

                        if map_kwargs_per_cell is not None:
                            mean_scores_arr = np.array([np.squeeze(score_results[arena_size][animal][curr_target_cell_id][metric_type]) for curr_target_cell_id in pkg_cell_ids])
                            score_results[arena_size][animal][metric_type] = package_scores(mean_scores_arr,
                                                                                            cell_ids=pkg_cell_ids)
                        else:
                            score_results[arena_size][animal][metric_type] = package_scores(score_results[arena_size][animal][metric_type],
                                                                                            cell_ids=pkg_cell_ids)

    return score_results

def compare_model(dataset, model=None, model_resp=None,
                  cfg=None, arena_sizes=None,
                  spec_resp_agg=None,
                  num_stimuli_types=1,
                  map_type="identity", map_kwargs={},
                  train_frac=0.0, num_train_test_splits=10,
                  split_start_seed=0,
                  model_pred_layer="g",
                  n_avg=100, trajectory_seed=0, return_responses=False,
                  scorer=None, apply_sac_mask=False, shape_2d=None,
                  map_kwargs_per_cell=None):

    """Compare model features to animal with a given map"""

    if model_resp is None:
        assert cfg is not None

    if not isinstance(cfg, list):
        cfg = [cfg]

    if not isinstance(trajectory_seed, list):
        trajectory_seed = [trajectory_seed]

    score_results = {}
    for arena_size in arena_sizes:
        print(f"Comparing to animals with arenas of size {arena_size}")

        if model_resp is None:
            model_resp = []
            for curr_seed in trajectory_seed:
                for curr_cfg in cfg:
                    model_act = get_model_activations(dataset=dataset,
                                                           model=model,
                                                           cfg=curr_cfg,
                                                           num_stimuli_types=num_stimuli_types,
                                                           arena_size=arena_size,
                                                           n_avg=n_avg,
                                                           model_pred_layer=model_pred_layer,
                                                           trajectory_seed=curr_seed)
                    num_units = model_act.shape[-1]
                    model_act = model_act.reshape((-1, num_units))
                    model_resp.append(model_act)
            model_resp = np.concatenate(model_resp, axis=0)
        else:
            print("Using passed in model response")
            # only one saved out model response for each arena size
            assert(len(arena_sizes) == 1)
            assert(len(cfg) == 1)
            assert(len(trajectory_seed) == 1)
            assert(num_stimuli_types == 1)
            if len(model_resp.shape) == 3:
                num_units = model_resp.shape[-1]
                model_resp = model_resp.reshape((-1, num_units))
                print(f"Passed in model response reshaped to {model_resp.shape}")
            else:
                assert(len(model_resp.shape) == 2)

        score_results[arena_size] = {}
        arena_animals = list(spec_resp_agg[arena_size].keys())
        for target_animal in arena_animals:
            target_animal_resp = spec_resp_agg[arena_size][target_animal]["resp"]
            target_animal_cell_ids = spec_resp_agg[arena_size][target_animal]["cell_ids"]
            subroutine_kwargs = {"X": model_resp, "map_type": map_type,
                                 "num_train_test_splits": num_train_test_splits,
                                 "split_start_seed": split_start_seed,
                                 "return_responses": return_responses,
                                 "scorer": scorer, "apply_sac_mask": apply_sac_mask,
                                 "shape_2d": shape_2d}

            if map_kwargs_per_cell is not None:
                mean_scores = {}
                for n, curr_target_cell_id in enumerate(target_animal_cell_ids):
                    curr_subroutine_kwargs = copy.deepcopy(subroutine_kwargs)
                    if len(target_animal_resp.shape) == 3:
                        curr_subroutine_kwargs["Y"] = np.expand_dims(target_animal_resp[:, :, n], axis=-1)
                    elif len(target_animal_resp.shape) == 2:
                        curr_subroutine_kwargs["Y"] = np.expand_dims(target_animal_resp[:, n], axis=-1)
                    else:
                        raise ValueError
                    curr_subroutine_kwargs["map_kwargs"] = map_kwargs_per_cell[curr_target_cell_id]["map_kwargs"]
                    # can specify a train frac per cell, otherwise it globally uses the one passed in above
                    curr_subroutine_kwargs["train_frac"] = map_kwargs_per_cell[curr_target_cell_id].get("train_frac", train_frac)
                    curr_mean_scores = fit_subroutine(**curr_subroutine_kwargs)
                    mean_scores[curr_target_cell_id] = curr_mean_scores
            else:
                subroutine_kwargs["Y"] = target_animal_resp
                subroutine_kwargs["map_kwargs"] = map_kwargs
                subroutine_kwargs["train_frac"] = train_frac
                mean_scores = fit_subroutine(**subroutine_kwargs)

            metrics = ["corr"] if map_kwargs_per_cell is not None else mean_scores.keys()
            for metric_type in metrics:
                if metric_type not in ["resp", "dist"]:
                    if map_kwargs_per_cell is not None:
                        mean_scores_arr = np.array([np.squeeze(mean_scores[curr_target_cell_id][metric_type]) for curr_target_cell_id in target_animal_cell_ids])
                        mean_scores[metric_type] = package_scores(mean_scores_arr,
                                                                  cell_ids=target_animal_cell_ids)
                    else:
                        mean_scores[metric_type] = package_scores(mean_scores[metric_type],
                                                                  cell_ids=target_animal_cell_ids)

            score_results[arena_size][target_animal] = mean_scores

    return score_results

def _get_sac(scorer, resp, apply_sac_mask=False):
    """Computes spatial autocorrelogram for an nbins x nbins x num_cells resp."""
    assert(len(resp.shape)==3)
    resp_sacs = []
    for cell_idx in range(resp.shape[-1]):
        curr_sac = scorer.calculate_sac(resp[:, :, cell_idx])
        if apply_sac_mask:
            curr_sac = curr_sac * scorer._plotting_sac_mask
        resp_sacs.append(curr_sac)
    resp_sacs = np.stack(resp_sacs, axis=-1)
    assert(len(resp_sacs.shape) == 3)
    assert(resp_sacs.shape[-1] == resp.shape[-1])
    return resp_sacs

def _get_sac_spec_resp(spec_resp_agg, arena_sizes,
                       scorer, apply_sac_mask=False):
    # each grid scorer is adapted to given arena size
    assert((arena_sizes is not None) and len(arena_sizes) == 1)
    new_spec_resp_agg = {}
    for curr_arena in arena_sizes:
        new_spec_resp_agg[curr_arena] = {}
        for curr_animal in spec_resp_agg[curr_arena].keys():
            new_spec_resp_agg[curr_arena][curr_animal] = {}
            assert("resp" in spec_resp_agg[curr_arena][curr_animal].keys())
            for inner_key in spec_resp_agg[curr_arena][curr_animal].keys():
                if inner_key == "resp":
                    new_spec_resp_agg[curr_arena][curr_animal]["resp"] = _get_sac(scorer=scorer,
                                                                                resp=spec_resp_agg[curr_arena][curr_animal]["resp"],
                                                                                apply_sac_mask=apply_sac_mask)
                else:
                    new_spec_resp_agg[curr_arena][curr_animal][inner_key] = spec_resp_agg[curr_arena][curr_animal][inner_key]

    return new_spec_resp_agg

def get_fits(dataset, model=None, model_resp=None, cfg=None,
                  arena_sizes=None, spec_resp_agg=None,
                  num_stimuli_types=1,
                  map_type="identity", map_kwargs={},
                  train_frac=0.0, num_train_test_splits=10,
                  split_start_seed=0,
                  n_avg=100, smooth_std=1,
                  interanimal_mode="pairwise",
                  trajectory_seed=0,
                  fit_place_cell_resp=False,
                  weights_save_dir=None,
                  return_responses=False,
                  model_pred_layer="g",
                  scorer=None, apply_sac_mask=False, shape_2d=None,
                  source_cell_ids=None,
                  target_cell_ids=None,
                  map_kwargs_per_cell=None):

    if arena_sizes is None:
        arena_sizes = list(np.unique(dataset["arena_size_cm"]))
    # aggregate neural responses per specimen
    if spec_resp_agg is None:
        if hasattr(dataset, 'spec_resp_agg'):
            print("Using native spec resp agg")
            spec_resp_agg = dataset.spec_resp_agg
        else:
            print("Aggregating neural responses")
            from mec_hpc_investigations.neural_data.utils import aggregate_responses
            spec_resp_agg = aggregate_responses(dataset=dataset,
                                                arena_sizes=arena_sizes,
                                                smooth_std=smooth_std)

    if ((model is None) and (model_resp is None)): # compare across animals
        place_cell_resp = None
        if fit_place_cell_resp:
            # can only do this for a single arena at a time
            assert(len(arena_sizes) == 1)
            arena_size = arena_sizes[0]
            place_cell_resp = get_model_activations(dataset=dataset,
                                               model="place_cells",
                                               cfg=cfg,
                                               arena_size=arena_size,
                                               n_avg=n_avg,
                                               trajectory_seed=trajectory_seed)

        score_results = compare_animals(dataset=dataset, arena_sizes=arena_sizes,
                                        spec_resp_agg=spec_resp_agg,
                                        map_type=map_type, map_kwargs=map_kwargs,
                                        train_frac=train_frac,
                                        num_train_test_splits=num_train_test_splits,
                                        split_start_seed=split_start_seed,
                                        mode=interanimal_mode,
                                        place_cell_resp=place_cell_resp,
                                        weights_save_dir=weights_save_dir,
                                        return_responses=return_responses,
                                        scorer=scorer, apply_sac_mask=apply_sac_mask,
                                        shape_2d=shape_2d,
                                        source_cell_ids=source_cell_ids,
                                        target_cell_ids=target_cell_ids,
                                        map_kwargs_per_cell=map_kwargs_per_cell)
    else:
        assert(fit_place_cell_resp is False)
        assert(weights_save_dir is None)
        assert(source_cell_ids is None)
        assert(target_cell_ids is None)
        score_results = compare_model(dataset=dataset,
                                      model=model, model_resp=model_resp,
                                      cfg=cfg, arena_sizes=arena_sizes,
                                      spec_resp_agg=spec_resp_agg,
                                      num_stimuli_types=num_stimuli_types,
                                      map_type=map_type,
                                      map_kwargs=map_kwargs,
                                      train_frac=train_frac,
                                      num_train_test_splits=num_train_test_splits,
                                      split_start_seed=split_start_seed,
                                      n_avg=n_avg,
                                      trajectory_seed=trajectory_seed,
                                      return_responses=return_responses,
                                      model_pred_layer=model_pred_layer,
                                      scorer=scorer, apply_sac_mask=apply_sac_mask,
                                      shape_2d=shape_2d,
                                      map_kwargs_per_cell=map_kwargs_per_cell)

    return score_results
