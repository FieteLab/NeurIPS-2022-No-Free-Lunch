import os
import itertools
import numpy as np
from mec_hpc_investigations.neural_data.remap_utils import aggregate_responses_1dvr
from mec_hpc_investigations.core.utils import get_params_from_workernum
from mec_hpc_investigations.core.default_dirs import CAITLIN1D_VR_INTERANIMAL_CON_RESULTS, BASE_DIR_PACKAGED, BASE_DIR_MODELS
from mec_hpc_investigations.neural_fits.utils import generate_train_test_splits
from mec_hpc_investigations.neural_data.metrics import noise_estimation
from mec_hpc_investigations.neural_fits.interanimal_utils import filename_constructor, construct_holdout_sources, load_sel_mask

def build_param_lookup(source_map_kwargs=None,
                       metric='pearsonr',
                       num_train_test_splits=10,
                       n_iter=900,
                       train_frac=0.5,
                       mode='pairwise',
                       correction='spearman_brown_split_half_denominator',
                       agg_resp_kwargs={},
                       sel_mask_name=None,
                       first_N_name=None,
                       n_jobs=5):

    spec_resp_agg = aggregate_responses_1dvr(**agg_resp_kwargs)
    sel_mask = None
    if sel_mask_name is not None:
        sel_mask = load_sel_mask(sel_mask_name)

    # build param lookup
    animals = list(spec_resp_agg.keys())
    param_lookup = {}
    key = 0

    if mode == "internal":
        assert(source_map_kwargs is None)
        assert(first_N_name is None)
        # that's only for the interanimal stuff
        assert(correction != 'spearman_brown_split_half_denominator')
        for target_animal in animals:
            for target_sess, target_sess_maps in spec_resp_agg[target_animal].items():
                for target_sess_map_idx, target_sess_map in enumerate(target_sess_maps):
                    if sel_mask is not None:
                        target_N = target_sess_map[:, :, sel_mask[target_animal][target_sess]]
                    else:
                        target_N = target_sess_map
                    param_lookup[str(key)] = {'source_N': None,
                                               'target_N': target_N,
                                               'source_map_kwargs': source_map_kwargs,
                                               'source_animal_name': target_animal,
                                               'source_session_name': target_sess,
                                               'source_map_name': target_sess_map_idx,
                                               'target_animal_name': target_animal,
                                               'target_session_name': target_sess,
                                               'target_map_name': target_sess_map_idx,
                                               'data_config': agg_resp_kwargs,
                                               'metric': metric,
                                               'num_train_test_splits': num_train_test_splits,
                                               'n_iter': n_iter,
                                               'train_frac': train_frac,
                                               'correction': correction,
                                               'sel_mask_name': sel_mask_name,
                                               'n_jobs': n_jobs
                                               }
                    key += 1

    elif mode == "pairwise":
        for animal_pair in itertools.permutations(animals, r=2):
            source_animal = animal_pair[0]
            target_animal = animal_pair[1]
            for target_sess, target_sess_maps in spec_resp_agg[target_animal].items():
                for target_sess_map_idx, target_sess_map in enumerate(target_sess_maps):
                    for source_sess, source_sess_maps in spec_resp_agg[source_animal].items():
                        for source_sess_map_idx, source_sess_map in enumerate(source_sess_maps):
                            if sel_mask is not None:
                                source_N = source_sess_map[:, :, sel_mask[source_animal][source_sess]]
                                target_N = target_sess_map[:, :, sel_mask[target_animal][target_sess]]
                            else:
                                source_N = source_sess_map
                                target_N = target_sess_map

                            param_lookup[str(key)] = {'source_N': source_N,
                                                       'target_N': target_N,
                                                       'source_map_kwargs': source_map_kwargs,
                                                       'source_animal_name': source_animal,
                                                       'source_session_name': source_sess,
                                                       'source_map_name': source_sess_map_idx,
                                                       'target_animal_name': target_animal,
                                                       'target_session_name': target_sess,
                                                       'target_map_name': target_sess_map_idx,
                                                       'data_config': agg_resp_kwargs,
                                                       'metric': metric,
                                                       'num_train_test_splits': num_train_test_splits,
                                                       'n_iter': n_iter,
                                                       'train_frac': train_frac,
                                                       'correction': correction,
                                                       'sel_mask_name': sel_mask_name,
                                                       'first_N_name': first_N_name,
                                                       'n_jobs': n_jobs
                                                       }
                            key += 1

    elif mode == "holdout":
        for target_animal in animals:
            holdout_sources = construct_holdout_sources(spec_resp_agg=spec_resp_agg,
                                                        target_animal=target_animal,
                                                        sel_mask_name=sel_mask_name)
            for target_sess, target_sess_maps in spec_resp_agg[target_animal].items():
                for target_sess_map_idx, target_sess_map in enumerate(target_sess_maps):
                    for holdout_source_idx, holdout_source_resp in enumerate(holdout_sources):
                        if sel_mask is not None:
                            target_N = target_sess_map[:, :, sel_mask[target_animal][target_sess]]
                        else:
                            target_N = target_sess_map
                        param_lookup[str(key)] = {'source_N': holdout_source_resp,
                                                   'target_N': target_N,
                                                   'source_map_kwargs': source_map_kwargs,
                                                   'source_animal_name': "holdout",
                                                   'source_session_name': 0, # as though there is 1 holdout recording
                                                   'source_map_name': holdout_source_idx,
                                                   'target_animal_name': target_animal,
                                                   'target_session_name': target_sess,
                                                   'target_map_name': target_sess_map_idx,
                                                   'data_config': agg_resp_kwargs,
                                                   'metric': metric,
                                                   'num_train_test_splits': num_train_test_splits,
                                                   'n_iter': n_iter,
                                                   'train_frac': train_frac,
                                                   'correction': correction,
                                                   'sel_mask_name': sel_mask_name,
                                                   'first_N_name': first_N_name,
                                                   'n_jobs': n_jobs
                                                   }
                        key += 1
    else:
        raise ValueError


    return param_lookup


def compute_interanimal_consistencies(source_N,
                                      target_N,
                                      source_map_kwargs,
                                      source_animal_name,
                                      source_session_name,
                                      source_map_name,
                                      target_animal_name,
                                      target_session_name,
                                      target_map_name,
                                      data_config,
                                      metric='pearsonr',
                                      num_train_test_splits=10,
                                      n_iter=900,
                                      train_frac=0.5,
                                      correction='spearman_brown_split_half_denominator',
                                      sel_mask_name=None,
                                      first_N_name=None,
                                      n_jobs=5):

    assert(len(target_N.shape) == 3)
    if source_N is not None:
        assert(source_map_kwargs is not None)
        assert(len(source_N.shape) == 3)
        assert(source_N.shape[1] == target_N.shape[1])

    first_N = None
    if first_N_name is not None:
        assert(source_N is not None)
        first_N = np.load(os.path.join(BASE_DIR_MODELS, first_N_name + ".npz"), allow_pickle=True)['arr_0'][()]
        # number of states between first and source should be equal
        if len(first_N.shape) == 2:
            assert(first_N.shape[0] == source_N.shape[1])
        elif len(first_N.shape) == 3:
            assert(first_N.shape[1] == source_N.shape[1])
        else:
            raise ValueError

    agg_results = []
    train_test_splits = generate_train_test_splits(num_states=target_N.shape[1], # number of position bins
                                                       num_splits=num_train_test_splits,
                                                       train_frac=train_frac)
    for curr_sp in train_test_splits:
        if source_N is None:
            assert(source_map_kwargs is None)
            curr_results = noise_estimation(target_N=target_N[:,curr_sp['test'],:],
                                            parallelize_per_target_unit=False,
                                            metric=metric,
                                            mode=correction,
                                            center=np.nanmean,
                                            summary_center='raw',
                                            sync=True,
                                            n_iter=n_iter,
                                            n_jobs=n_jobs)
        else:
            curr_results = noise_estimation(target_N=target_N,
                                            source_N=source_N,
                                            source_map_kwargs=source_map_kwargs,
                                            first_N=first_N,
                                            parallelize_per_target_unit=False,
                                            train_img_idx=curr_sp['train'], test_img_idx=curr_sp['test'],
                                            metric=metric,
                                            mode=correction,
                                            center=np.nanmean,
                                            summary_center='raw',
                                            sync=True,
                                            n_iter=n_iter,
                                            n_jobs=n_jobs)

        curr_results = np.expand_dims(curr_results, axis=0)
        agg_results.append(curr_results)
    agg_results = np.concatenate(agg_results, axis=0) # (num_train_test_splits, num_bs_trials, num_target_units)

    filename = filename_constructor(dataset='caitlin1dvr',
                                      source_map_kwargs=source_map_kwargs,
                                      source_animal_name=source_animal_name,
                                      source_session_name=source_session_name,
                                      source_map_name=source_map_name,
                                      target_animal_name=target_animal_name,
                                      target_session_name=target_session_name,
                                      target_map_name=target_map_name,
                                      data_config=data_config,
                                      metric=metric,
                                      num_train_test_splits=num_train_test_splits,
                                      n_iter=n_iter,
                                      train_frac=train_frac,
                                      correction=correction,
                                      sel_mask_name=sel_mask_name,
                                      first_N_name=first_N_name,
                                      file_ext='.npz')
    filename = os.path.join(CAITLIN1D_VR_INTERANIMAL_CON_RESULTS, filename)
    np.savez(filename, agg_results)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_type", type=str, default="percentile")
    parser.add_argument("--percentile", type=float, default=75)
    parser.add_argument("--percentile_identity", type=bool, default=False)
    parser.add_argument("--pls_n_components", type=int, default=9)
    parser.add_argument("--pls_fit_per_target_unit", type=bool, default=False)
    parser.add_argument("--mode", type=str, default="pairwise", choices=["internal", "pairwise", "holdout"])
    parser.add_argument("--sel_mask_name", type=str, default=None)
    parser.add_argument("--first_N_name", type=str, default=None)
    parser.add_argument("--n_jobs", type=int, default=5)
    args = parser.parse_args()

    print("Looking up params...")
    if args.mode == "internal":
        assert(args.first_N_name is None)
        param_lookup = build_param_lookup(
            metric='pearsonr',
            mode=args.mode,
            correction='spearman_brown_split_half',
            sel_mask_name=args.sel_mask_name,
            n_jobs=args.n_jobs
        )
    else:
        if args.map_type == "percentile":
            source_map_kwargs = {'map_type': 'percentile', 'map_kwargs': {'identity': args.percentile_identity, 'percentile': args.percentile}}
            param_lookup = build_param_lookup(
                source_map_kwargs=source_map_kwargs,
                metric='pearsonr',
                mode=args.mode,
                sel_mask_name=args.sel_mask_name,
                first_N_name=args.first_N_name,
                n_jobs=args.n_jobs
            )
        elif args.map_type == "pls":
            source_map_kwargs = {'map_type': 'pls', 'map_kwargs': {'n_components': args.pls_n_components, 'fit_per_target_unit': args.pls_fit_per_target_unit}}
            param_lookup = build_param_lookup(
                source_map_kwargs=source_map_kwargs,
                metric='pearsonr',
                mode=args.mode,
                sel_mask_name=args.sel_mask_name,
                first_N_name=args.first_N_name,
                n_jobs=args.n_jobs
            )
        elif args.map_type == "rsa":
            source_map_kwargs = {'map_type': 'identity', 'map_kwargs': {}}
            param_lookup = build_param_lookup(
                source_map_kwargs=source_map_kwargs,
                metric='rsa',
                mode=args.mode,
                sel_mask_name=args.sel_mask_name,
                first_N_name=args.first_N_name,
                n_jobs=args.n_jobs
            )
        else:
            raise ValueError(f"{args.map_type} not implemented yet.")

    print("NUM TOTAL JOBS", len(list(param_lookup.keys())))
    curr_params = get_params_from_workernum(worker_num=os.environ.get('SLURM_ARRAY_TASK_ID'),
                                            param_lookup=param_lookup)
    print("CURR PARAMS", curr_params)
    compute_interanimal_consistencies(**curr_params)
