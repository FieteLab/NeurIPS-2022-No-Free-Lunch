import numpy as np
import xarray as xr
import os
from scipy.stats import sem
from mec_hpc_investigations.core.utils import dict_to_str
from mec_hpc_investigations.neural_fits.utils import package_scores
from mec_hpc_investigations.neural_data.remap_utils import aggregate_responses_1dvr, pad_1dvr_trials
from mec_hpc_investigations.core.default_dirs import CAITLIN1D_VR_INTERANIMAL_CON_RESULTS, BASE_DIR_RESULTS, BASE_DIR_PACKAGED

def filename_constructor(source_map_kwargs=None,
                          source_animal_name=None,
                          source_session_name=None,
                          source_map_name=None,
                          target_animal_name=None,
                          target_session_name=None,
                          target_map_name=None,
                          prefix="interancons_",
                          dataset='caitlin1dvr',
                          data_config={},
                          metric='pearsonr',
                          num_train_test_splits=10,
                          n_iter=900,
                          train_frac=0.5,
                          correction='spearman_brown_split_half_denominator',
                          sel_mask_name=None,
                          first_N_name=None,
                          model_pred_layer=None,
                          file_ext='.npz'):

    file_nm = prefix + f"{dataset}_dataconfig{dict_to_str(data_config)}"
    if source_map_kwargs is not None:
        file_nm += f"_{dict_to_str(source_map_kwargs)}"
    else:
        file_nm += "_sourcemapNone"
    file_nm += f"_{train_frac}"
    file_nm += f"_sp{num_train_test_splits}"

    if first_N_name is not None:
        file_nm += f"_first_{first_N_name}"
    if source_animal_name is not None:
        assert((source_session_name is not None) and (source_map_name is not None))
        file_nm += f"_source_{source_animal_name}_{source_session_name}_{source_map_name}"
    if model_pred_layer is not None:
        file_nm += f"_layer{model_pred_layer}"
    if target_animal_name is not None:
        assert((target_session_name is not None) and (target_map_name is not None))
        file_nm += f"_target_{target_animal_name}_{target_session_name}_{target_map_name}"

    file_nm += f"_{metric}"
    file_nm += f"_niter{n_iter}"
    file_nm += f"_{correction}"

    if sel_mask_name is not None:
        file_nm += f"_{sel_mask_name}"

    if file_ext is not None:
        file_nm += file_ext

    return file_nm

def load_sel_mask(sel_mask_name):
    sel_mask = np.load(os.path.join(BASE_DIR_PACKAGED, sel_mask_name + ".npz"), allow_pickle=True)['arr_0'][()]
    return sel_mask

def construct_holdout_sources(spec_resp_agg,
                              target_animal,
                              sel_mask_name=None):
    if sel_mask_name is not None:
        sel_mask = load_sel_mask(sel_mask_name)
    else:
        sel_mask = None
    animals = list(spec_resp_agg.keys())
    assert(target_animal in animals)
    holdout_source_resp = {}
    for source_animal in animals:
        # combine all of the sessions that are not the target animal
        if source_animal != target_animal:
            for sess, sess_resp in spec_resp_agg[source_animal].items():
                # sanity check that each session is distinct to avoid overwriting
                assert(sess not in holdout_source_resp.keys())
                if sel_mask is not None:
                    assert(len(sess_resp.shape) == 3)
                    holdout_source_resp[sess] = sess_resp[:, :, sel_mask[source_animal][sess]]
                else:
                    holdout_source_resp[sess] = sess_resp

    # concatenate maps across source animals & their sessions (neural population), one map each per source animal & recording session
    holdout_sources = [pad_1dvr_trials(list(values)) for values in itertools.product(*holdout_source_resp.values())]
    return holdout_sources

def agg_interanimal_consistencies(source_map_kwargs,
                                  spec_resp_agg=None,
                                  model_name=None,
                                  mode='pairwise',
                                  dataset='caitlin1dvr',
                                  agg_resp_kwargs={},
                                  metric='pearsonr',
                                  num_train_test_splits=10,
                                  n_iter=900,
                                  train_frac=0.5,
                                  correction='spearman_brown_split_half_denominator',
                                  file_ext='.npz',
                                  trial_agg_func=np.nanmean,
                                  train_test_agg_func=np.mean,
                                  source_agg_func=np.mean,
                                  target_agg_func=np.mean,
                                  sel_mask=None,
                                  sel_mask_name=None,
                                  first_N_name=None,
                                  save=True,
                                  dir_path=None,
                                  save_path=None,
                                  model_pred_layer=None
                                  ):

    if save_path is None:
        save_path = BASE_DIR_RESULTS
    if dataset == 'caitlin1dvr':
        if spec_resp_agg is None:
            spec_resp_agg = aggregate_responses_1dvr(**agg_resp_kwargs)
        if dir_path is None:
            if model_name is None:
                dir_path = CAITLIN1D_VR_INTERANIMAL_CON_RESULTS
            else:
                dir_path = BASE_DIR_RESULTS
    else:
        raise ValueError

    if sel_mask_name is not None:
        # use this option if you computed the fits with a selection mask
        assert(sel_mask is None)
        sel_mask = load_sel_mask(sel_mask_name)
    elif sel_mask is not None:
        # use this option to post-select after the fact (available for only some situations)
        assert(sel_mask_name is None)

    animals = list(spec_resp_agg.keys())
    interanimal_cons = []
    interanimal_cons_sem = []
    num_files = 0
    for target_animal in animals:
        if mode == "holdout":
            holdout_sources = construct_holdout_sources(spec_resp_agg=spec_resp_agg,
                                                        target_animal=target_animal,
                                                        sel_mask_name=sel_mask_name)
        for target_sess, target_sess_maps in spec_resp_agg[target_animal].items():
            target_map_avg_cons = []
            for target_sess_map_idx, target_sess_map in enumerate(target_sess_maps):
                source_avg_cons = []
                if model_name is not None:
                    fn = filename_constructor(source_map_kwargs=source_map_kwargs,
                                              source_animal_name=model_name,
                                              source_session_name=0,
                                              source_map_name=0,
                                              target_animal_name=target_animal,
                                              target_session_name=target_sess,
                                              target_map_name=target_sess_map_idx,
                                              dataset=dataset,
                                              prefix="",
                                              data_config=agg_resp_kwargs,
                                              metric=metric,
                                              num_train_test_splits=num_train_test_splits,
                                              n_iter=n_iter,
                                              train_frac=train_frac,
                                              correction=correction,
                                              sel_mask_name=sel_mask_name,
                                              file_ext=file_ext)

                    if model_pred_layer is not None:
                        curr_agg_results = np.load(os.path.join(dir_path, fn), allow_pickle=True)['arr_0'][()][model_pred_layer]
                    else:
                        curr_agg_results = np.load(os.path.join(dir_path, fn), allow_pickle=True)['arr_0'][()]
                    assert(len(curr_agg_results.shape) == 3)
                    if (sel_mask is not None) and (sel_mask_name is None):
                        # this indicates we post-select since model was not trained with selection mask
                        assert(metric != "rsa")
                        assert(source_map_kwargs["map_type"] != "pls")
                        print("Selecting out target units")
                        curr_agg_results = curr_agg_results[:, :, sel_mask[target_animal][target_sess]]
                        assert(len(curr_agg_results.shape) == 3)
                    # average across train/test splits and bootstrap trials
                    curr_cons = train_test_agg_func(trial_agg_func(curr_agg_results, axis=1), axis=0)
                    source_avg_cons.append(curr_cons)
                    num_files += 1
                else:
                    assert(model_pred_layer is None)
                    if mode == "internal":
                        assert(source_map_kwargs is None)
                        assert(first_N_name is None)
                        # that's only for the interanimal stuff
                        assert(correction != 'spearman_brown_split_half_denominator')
                        fn = filename_constructor(source_map_kwargs=source_map_kwargs,
                                                  source_animal_name=target_animal,
                                                  source_session_name=target_sess,
                                                  source_map_name=target_sess_map_idx,
                                                  target_animal_name=target_animal,
                                                  target_session_name=target_sess,
                                                  target_map_name=target_sess_map_idx,
                                                  dataset=dataset,
                                                  data_config=agg_resp_kwargs,
                                                  metric=metric,
                                                  num_train_test_splits=num_train_test_splits,
                                                  n_iter=n_iter,
                                                  train_frac=train_frac,
                                                  correction=correction,
                                                  sel_mask_name=sel_mask_name,
                                                  file_ext=file_ext)
                        curr_agg_results = np.load(os.path.join(dir_path, fn), allow_pickle=True)['arr_0'][()]
                        assert(len(curr_agg_results.shape) == 3)
                        if (sel_mask is not None) and (sel_mask_name is None):
                            # this indicates we post-select since model was not trained with selection mask
                            assert(metric != "rsa")
                            assert(source_map_kwargs["map_type"] != "pls")
                            print("Selecting out target units")
                            curr_agg_results = curr_agg_results[:, :, sel_mask[target_animal][target_sess]]
                            assert(len(curr_agg_results.shape) == 3)
                        # average across train/test splits and bootstrap trials
                        curr_cons = train_test_agg_func(trial_agg_func(curr_agg_results, axis=1), axis=0)
                        source_avg_cons.append(curr_cons)
                        num_files += 1
                    elif mode == "pairwise":
                        # if you want to subselect neurons here you need to recompute the whole quantity to select source units out too
                        if sel_mask_name is None:
                            assert(sel_mask is None)
                        for source_animal in animals:
                            if source_animal != target_animal:
                                for source_sess, source_sess_maps in spec_resp_agg[source_animal].items():
                                    for source_sess_map_idx, source_sess_map in enumerate(source_sess_maps):
                                        fn = filename_constructor(source_map_kwargs=source_map_kwargs,
                                                                  source_animal_name=source_animal,
                                                                  source_session_name=source_sess,
                                                                  source_map_name=source_sess_map_idx,
                                                                  target_animal_name=target_animal,
                                                                  target_session_name=target_sess,
                                                                  target_map_name=target_sess_map_idx,
                                                                  dataset=dataset,
                                                                  data_config=agg_resp_kwargs,
                                                                  metric=metric,
                                                                  num_train_test_splits=num_train_test_splits,
                                                                  n_iter=n_iter,
                                                                  train_frac=train_frac,
                                                                  correction=correction,
                                                                  sel_mask_name=sel_mask_name,
                                                                  first_N_name=first_N_name,
                                                                  file_ext=file_ext)

                                        curr_agg_results = np.load(os.path.join(dir_path, fn), allow_pickle=True)['arr_0'][()]
                                        assert(len(curr_agg_results.shape) == 3)
                                        # average across train/test splits and bootstrap trials
                                        curr_cons = train_test_agg_func(trial_agg_func(curr_agg_results, axis=1), axis=0)
                                        source_avg_cons.append(curr_cons)
                                        num_files += 1
                    elif mode == "holdout":
                        # if you want to subselect neurons here you need to recompute the whole quantity to select source units out too
                        if sel_mask_name is None:
                            assert(sel_mask is None)
                        for holdout_source_idx, holdout_source_resp in enumerate(holdout_sources):
                            fn = filename_constructor(source_map_kwargs=source_map_kwargs,
                                                      source_animal_name="holdout",
                                                      source_session_name=0,
                                                      source_map_name=holdout_source_idx,
                                                      target_animal_name=target_animal,
                                                      target_session_name=target_sess,
                                                      target_map_name=target_sess_map_idx,
                                                      dataset=dataset,
                                                      data_config=agg_resp_kwargs,
                                                      metric=metric,
                                                      num_train_test_splits=num_train_test_splits,
                                                      n_iter=n_iter,
                                                      train_frac=train_frac,
                                                      correction=correction,
                                                      sel_mask_name=sel_mask_name,
                                                      first_N_name=first_N_name,
                                                      file_ext=file_ext)

                            curr_agg_results = np.load(os.path.join(dir_path, fn), allow_pickle=True)['arr_0'][()]
                            assert(len(curr_agg_results.shape) == 3)
                            # average across train/test splits and bootstrap trials
                            curr_cons = train_test_agg_func(trial_agg_func(curr_agg_results, axis=1), axis=0)
                            source_avg_cons.append(curr_cons)
                            num_files += 1
                    else:
                        raise ValueError

                # average across sources
                source_avg_cons = source_agg_func(np.stack(source_avg_cons, axis=0), axis=0)
                target_map_avg_cons.append(source_avg_cons)

            # average consistency across target maps
            target_map_avg_cons_agg = np.stack(target_map_avg_cons, axis=0)
            target_map_avg_cons = target_agg_func(target_map_avg_cons_agg, axis=0)
            interanimal_cons.append(target_map_avg_cons)
            # sem consistency across target maps
            target_map_sem_cons = sem(target_map_avg_cons_agg, axis=0, nan_policy="omit")
            interanimal_cons_sem.append(target_map_sem_cons)

    # concatenate across distinct neural populations (animals & their recording session)
    interanimal_cons = np.concatenate(interanimal_cons, axis=-1)
    interanimal_cons_sem = np.concatenate(interanimal_cons_sem, axis=-1)

    ret_dict = {"mean": interanimal_cons, "sem": interanimal_cons_sem}

    print(f"Finished processing {num_files} files")

    # TODO: call package scores once you package cell ids with the binned data, and feed it to spec_resp_agg

    if save:
        save_filename = filename_constructor(source_map_kwargs=source_map_kwargs,
                                              prefix="interancons_" if model_name is None else f"{model_name}_",
                                              dataset=dataset + f"_mode{mode}" if model_name is None else dataset,
                                              data_config=agg_resp_kwargs,
                                              metric=metric,
                                              num_train_test_splits=num_train_test_splits,
                                              n_iter=n_iter,
                                              train_frac=train_frac,
                                              correction=correction,
                                              sel_mask_name=sel_mask_name,
                                              first_N_name=first_N_name,
                                              model_pred_layer=model_pred_layer,
                                              file_ext=file_ext)
        save_fp = os.path.join(save_path, save_filename)
        np.savez(save_fp, ret_dict)
        print(f"Saved to {save_fp}")

    return ret_dict
