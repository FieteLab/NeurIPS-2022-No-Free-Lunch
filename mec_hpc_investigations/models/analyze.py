import joblib
import numpy as np
import os
import pandas as pd
from typing import Dict, List, Tuple, Union
import wandb


def compute_minima_performance_metrics_from_runs_histories(runs_histories_df: pd.DataFrame):
    minima_performance_metrics = runs_histories_df.groupby(['run_id']).agg({
        'max_grid_score_d=60_n=256': 'max',
        'max_grid_score_d=90_n=256': 'max',
        'pos_decoding_err': 'min',
        'loss': 'min',
        'participation_ratio': 'max',
    })

    # Convert run_id from index to column.
    minima_performance_metrics.reset_index(inplace=True)

    return minima_performance_metrics


def download_wandb_project_runs_configs(wandb_project_path: str,
                                        data_dir: str,
                                        sweep_ids: List[str] = None,
                                        finished_only: bool = True,
                                        refresh: bool = False,
                                        ) -> pd.DataFrame:
    runs_configs_df_path = os.path.join(
        data_dir,
        'sweeps=' + ','.join(sweep_ids) + '_runs_configs.csv')
    if refresh or not os.path.isfile(runs_configs_df_path):

        # Download sweep results
        api = wandb.Api(timeout=60)

        # Project is specified by <entity/project-name>
        if sweep_ids is None:
            runs = api.runs(path=wandb_project_path)
        else:
            runs = []
            for sweep_id in sweep_ids:
                runs.extend(api.runs(path=wandb_project_path,
                                     filters={"Sweep": sweep_id}))

        sweep_results_list = []
        for run in runs:
            # .summary contains the output keys/values for metrics like accuracy.
            #  We call ._json_dict to omit large files
            summary = run.summary._json_dict

            # .config contains the hyperparameters.
            #  We remove special values that start with _.
            summary.update(
                {k: v for k, v in run.config.items()
                 if not k.startswith('_')})

            summary.update({'State': run.state,
                            'Sweep': run.sweep.id if run.sweep is not None else None,
                            'run_id': run.id})
            # .name is the human-readable name of the run.
            summary.update({'run_name': run.name})
            sweep_results_list.append(summary)

        runs_configs_df = pd.DataFrame(sweep_results_list)
        runs_configs_df.to_csv(runs_configs_df_path, index=False)
        print(f'Wrote {runs_configs_df_path} to disk.')
    else:
        runs_configs_df = pd.read_csv(runs_configs_df_path)
        print(f'Loaded {runs_configs_df_path} from disk.')

    # Keep only finished runs
    finished_runs = runs_configs_df['State'] == 'finished'
    print(f"% of successfully finished runs: {finished_runs.mean()} ({finished_runs.sum()} / {len(finished_runs)})")

    if finished_only:
        runs_configs_df = runs_configs_df[finished_runs]

        # Check that we don't have an empty data frame.
        assert len(runs_configs_df) > 0

        # Ensure we aren't working with a slice.
        runs_configs_df = runs_configs_df.copy()

    return runs_configs_df


def download_wandb_project_runs_histories(wandb_project_path: str,
                                          data_dir: str,
                                          sweep_ids: List[str] = None,
                                          num_samples: int = 10000,
                                          refresh: bool = False,
                                          keys: List[str] = None,
                                          ) -> pd.DataFrame:
    if keys is None:
        keys = ['max_grid_score_d=60_n=256',
                'max_grid_score_d=90_n=256',
                'pos_decoding_err',
                'participation_ratio',
                'loss',
                'num_grad_steps',
                ]

    runs_histories_df_path = os.path.join(
        data_dir,
        'sweeps=' + ','.join(sweep_ids) + '_runs_histories.csv')
    if refresh or not os.path.isfile(runs_histories_df_path):

        # Download sweep results
        api = wandb.Api(timeout=60)

        # Project is specified by <entity/project-name>
        if sweep_ids is None:
            runs = api.runs(path=wandb_project_path)
        else:
            runs = []
            for sweep_id in sweep_ids:
                runs.extend(api.runs(path=wandb_project_path,
                                     filters={"Sweep": sweep_id}))

        runs_histories_list = []
        for run in runs:
            run_history_df = run.history(
                samples=num_samples,
                keys=keys)
            if len(run_history_df) == 0:
                continue
            run_history_df['run_id'] = run.id
            runs_histories_list.append(run_history_df)

        assert len(runs_histories_list) > 0
        runs_histories_df = pd.concat(runs_histories_list, sort=False)

        runs_histories_df.sort_values(
            ['run_id', '_step'],
            ascending=True,
            inplace=True)

        runs_histories_df['loss_over_min_loss'] = runs_histories_df.groupby('run_id')['loss'].apply(
            lambda col: col / np.min(col)
        )
        runs_histories_df.reset_index(inplace=True, drop=True)

        runs_histories_df['pos_decoding_err_over_min_pos_decoding_err'] = runs_histories_df.groupby('run_id')[
            'pos_decoding_err'].apply(
            lambda col: col / np.min(col)
        )
        runs_histories_df.reset_index(inplace=True, drop=True)

        runs_histories_df.to_csv(runs_histories_df_path, index=False)
        print(f'Wrote {runs_histories_df_path} to disk')
    else:
        runs_histories_df = pd.read_csv(runs_histories_df_path)
        print(f'Loaded {runs_histories_df_path} from disk.')

    return runs_histories_df


def open_run_joblib_files(run_ids: List[str],
                          results_dir: str = 'results',
                          ) -> Dict[str, Dict[str, np.ndarray]]:

    joblib_files_data_by_run_id = {run_id: dict() for run_id in run_ids}

    for run_id in run_ids:

        run_dir = os.path.join(results_dir, run_id)

        # Extract rate maps and scores.
        rate_maps_and_scores_joblib_path = os.path.join(run_dir, 'rate_maps_and_scores.joblib')
        rate_maps_and_scores_results = joblib.load(rate_maps_and_scores_joblib_path)

        # Extract periods, period errors and orientations.
        period_and_orientation_joblib_path = os.path.join(run_dir, 'period_results_path.joblib')
        period_and_orientation_results = joblib.load(period_and_orientation_joblib_path)

        joblib_files_data_by_run_id[run_id] = {
            'rate_maps': rate_maps_and_scores_results['rate_maps'],
            'score_60_by_neuron': rate_maps_and_scores_results['score_60_by_neuron'],
            'score_90_by_neuron': rate_maps_and_scores_results['score_90_by_neuron'],
            'period_per_cell': period_and_orientation_results['period_per_cell'],
            'period_err_per_cell': period_and_orientation_results['period_err_per_cell'],
            'orientations_per_cell': period_and_orientation_results['orientations_per_cell'],
        }

    return joblib_files_data_by_run_id
