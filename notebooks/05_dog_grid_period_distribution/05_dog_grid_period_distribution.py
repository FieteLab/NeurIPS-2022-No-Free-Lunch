import os

from mec_hpc_investigations.models.analyze import compute_minima_performance_metrics_from_runs_histories, \
    download_wandb_project_runs_configs, download_wandb_project_runs_histories
from mec_hpc_investigations.models.plot import *

# Declare
notebook_dir = 'notebooks/05_dog_grid_period_distribution'
data_dir = os.path.join(notebook_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
results_dir = os.path.join(notebook_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

low_pos_decoding_err_threshold_in_cm = 6.
grid_score_d60_threshold = 0.8
grid_score_d90_threshold = 1.5
sweep_ids = [
    'r83jf81o',  # 20: DoG+Global+CE good grid cells, with 2 sizes of receptive fields & 2 optimizers & 2 seeds
]

runs_configs_df = download_wandb_project_runs_configs(
    wandb_project_path='mec-hpc-investigations',
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    finished_only=True)


def add_human_readable_run_id(row: pd.Series):
    run_group = f"Opt={row['optimizer']}\nRF={row['place_cell_rf']}\nSeed={row['seed']}"
    return run_group


runs_configs_df['human_readable_run_id'] = runs_configs_df.apply(
    add_human_readable_run_id,
    axis=1)

# Keep only low position decoding errors.
runs_configs_df = runs_configs_df[runs_configs_df['pos_decoding_err'] < low_pos_decoding_err_threshold_in_cm]

runs_histories_df = download_wandb_project_runs_histories(
    wandb_project_path='mec-hpc-investigations',
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    keys=['max_grid_score_d=60_n=256',
          'max_grid_score_d=90_n=256',
          'pos_decoding_err',
          'participation_ratio',
          'loss',
          'num_grad_steps',
          'grid_score_histogram_d=60_n=256',
          'grid_score_histogram_d=90_n=256',
          ],
    refresh=True)

runs_augmented_histories_df = runs_configs_df[[
    'run_id', 'human_readable_run_id', 'optimizer', 'place_cell_rf', 'seed']].merge(
    runs_histories_df,
    on='run_id',
    how='left')

plot_grid_scores_histograms_by_run_id(
    runs_augmented_histories_df=runs_augmented_histories_df,
    plot_dir=results_dir,
)

plot_grid_periods_histograms_by_run_id(

)

# plot_grid_period_distribution()

print('Finished!')
