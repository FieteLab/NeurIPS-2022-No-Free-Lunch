import os
import numpy as np
import itertools
from mec_hpc_investigations.core.default_dirs import CAITLIN2D_INTERANIMAL_SAMPLE_RESULTS, CAITLIN2D_INTERANIMAL_SAMPLE_AGG_RESULTS
from mec_hpc_investigations.neural_fits.sample_elasticnet_per_neuron import construct_filename as ld_filename
from mec_hpc_investigations.core.utils import get_params_from_workernum
from mec_hpc_investigations.neural_data.datasets import CaitlinDatasetWithoutInertial
from mec_hpc_investigations.neural_data.utils import aggregate_responses

def build_param_lookup(train_frac,
                       arena_size=100,
                       smooth_std=1
                       ):

    dataset_obj = CaitlinDatasetWithoutInertial()
    dataset_obj.package_data()
    dataset = dataset_obj.packaged_data
    spec_resp_agg = aggregate_responses(dataset=dataset,
                                        smooth_std=smooth_std)
    arena_animals = list(spec_resp_agg[arena_size].keys())

    # build param lookup
    param_lookup = {}
    key = 0
    for target_animal in arena_animals:
        target_animal_resp = spec_resp_agg[arena_size][target_animal]["resp"]
        num_target_neurons = target_animal_resp.shape[-1]

        for n in list(range(num_target_neurons)):
            param_lookup[str(key)] = {
                                      "target_cell_id": spec_resp_agg[arena_size][target_animal]["cell_ids"][n],
                                      "train_frac": train_frac,
                                      }
            key += 1
    return param_lookup

def agg_per_neuron(target_cell_id, train_frac):
    m = np.load(os.path.join(CAITLIN2D_INTERANIMAL_SAMPLE_RESULTS, ld_filename(target_cell_id=target_cell_id)),
                allow_pickle=True)["arr_0"][()]
    sel_idx = np.where([m_e["train_frac"] == train_frac for m_e in m])[0]
    m_sel = m[sel_idx]

    train_scores = np.array([np.mean(m_e["train_scores"]) for m_e in m_sel])
    val_scores = np.array([np.mean(m_e["val_scores"]) for m_e in m_sel])
    test_scores = np.array([np.mean(m_e["test_scores"]) for m_e in m_sel])
    alphas = np.array([m_e["alpha"] for m_e in m_sel])
    l1_ratios = np.array([m_e["l1_ratio"] for m_e in m_sel])
    alpha_renorms = np.array([m_e["alpha_renorm"][0] for m_e in m_sel])
    # sanity check that alpha renorm is always the same across train/test splits
    for m_e in m_sel:
        for curr_alpha_renorm in m_e["alpha_renorm"]:
            assert(curr_alpha_renorm == m_e["alpha_renorm"][0])
    d = {"train_scores": train_scores, "val_scores": val_scores, "test_scores": test_scores, "alphas": alphas, "l1_ratios": l1_ratios, "alpha_renorms": alpha_renorms}

    return d

def construct_filename(target_cell_id,
                       train_frac,
                       dataset_name="caitlin2dwithoutinertial",
                       arena_size=100,
                       smooth_std=1,
                       num_train_test_splits=10,
                       num_cv_splits=3,
                       neural_map_str="percentile",
                  ):

    fname = "save_agg_per_neuron"
    fname += f"_trainfrac{train_frac}"
    fname += f"_dataset{dataset_name}"
    fname += f"_arenasize{arena_size}"
    fname += f"_smoothstd_{smooth_std}"
    fname += f"_targetcellid{target_cell_id}"
    fname += f"_maptype{neural_map_str}"
    fname += f"_numtrsp{num_train_test_splits}_numcvsp{num_cv_splits}"
    fname += ".npz"
    return fname

def save_results(fit_results,
                 target_cell_id,
                 train_frac,
                 dataset_name="caitlin2dwithoutinertial",
                  arena_size=100,
                  smooth_std=1,
                    num_train_test_splits=10,
                    num_cv_splits=3,
                    neural_map_str="percentile",
                    **kwargs
                  ):

    print(f"Saving results to this directory {CAITLIN2D_INTERANIMAL_SAMPLE_AGG_RESULTS}")
    fname = construct_filename(target_cell_id=target_cell_id,
                               train_frac=train_frac,
                             dataset_name=dataset_name,
                              arena_size=arena_size,
                              smooth_std=smooth_std,
                                num_train_test_splits=num_train_test_splits,
                                num_cv_splits=num_cv_splits,
                                neural_map_str=neural_map_str,
                  )

    np.savez(os.path.join(CAITLIN2D_INTERANIMAL_SAMPLE_AGG_RESULTS, fname), fit_results)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_frac", type=float, default=0.5)
    args = parser.parse_args()

    print('Looking up params')
    param_lookup = build_param_lookup(train_frac=args.train_frac)
    print('NUM JOBS', len(list(param_lookup.keys())))
    curr_params = get_params_from_workernum(worker_num=os.environ.get('SLURM_ARRAY_TASK_ID'), param_lookup=param_lookup)
    print('Curr params', curr_params)
    fit_results = agg_per_neuron(**curr_params)
    save_results(fit_results,
                 **curr_params)
