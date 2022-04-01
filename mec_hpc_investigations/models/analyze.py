import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union
import wandb


def compute_minima_performance_metrics_from_runs_histories(runs_histories_df: pd.DataFrame):

    minima_performance_metrics = runs_histories_df.groupby(['run_id']).agg({
        'max_grid_score_d=60_n=256': 'max',
        'max_grid_score_d=90_n=256': 'max',
        'pos_decoding_err': 'min',
        'loss': 'min',
    })

    # Convert run_id from index to column.
    minima_performance_metrics.reset_index(inplace=True)

    return minima_performance_metrics


def download_wandb_project_runs_configs(wandb_project_path: str,
                                        sweep_ids: List[str] = None,
                                        finished_only: bool = True,
                                        ) -> pd.DataFrame:
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

    sweep_results_df = pd.DataFrame(sweep_results_list)

    # Keep only finished runs
    finished_runs = sweep_results_df['State'] == 'finished'
    print(f"% of successfully finished runs: {finished_runs.mean()} ({finished_runs.sum() / len(finished_runs)})")

    if finished_only:
        sweep_results_df = sweep_results_df[finished_runs]

        # Check that we don't have an empty data frame.
        assert len(sweep_results_df) > 0

        # Ensure we aren't working with a slice.
        sweep_results_df = sweep_results_df.copy()

    return sweep_results_df


def download_wandb_project_runs_histories(wandb_project_path: str,
                                          sweep_ids: List[str] = None,
                                          num_samples: int = 10000,
                                          ) -> pd.DataFrame:

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
            keys=['max_grid_score_d=60_n=256',
                  'max_grid_score_d=90_n=256',
                  'pos_decoding_err',
                  'loss'])
        if len(run_history_df) == 0:
            continue
        run_history_df['run_id'] = run.id
        runs_histories_list.append(run_history_df)

    runs_histories_df = pd.concat(runs_histories_list, sort=False)

    runs_histories_df.sort_values(
        ['run_id', '_step'],
        ascending=True,
        inplace=True)

    runs_histories_df['loss_over_min_loss'] = runs_histories_df.groupby('run_id')['loss'].apply(
        lambda col: col / np.min(col)
    )

    runs_histories_df.reset_index(inplace=True, drop=True)

    runs_histories_df['pos_decoding_err_over_min_pos_decoding_err'] = runs_histories_df.groupby('run_id')['pos_decoding_err'].apply(
        lambda col: col / np.min(col)
    )

    runs_histories_df.reset_index(inplace=True, drop=True)

    return runs_histories_df

