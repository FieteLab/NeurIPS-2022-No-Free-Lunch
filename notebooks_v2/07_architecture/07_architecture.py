import os

from mec_hpc_investigations.models.analyze import compute_minima_performance_metrics_from_runs_histories, \
    download_wandb_project_runs_configs, download_wandb_project_runs_histories, load_runs_joblib_files
from mec_hpc_investigations.models.plot import *

# Declare variables.
notebook_dir = 'notebooks_v2/06_optimizer'
data_dir = os.path.join(notebook_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
results_dir = os.path.join(notebook_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

low_pos_decoding_err_threshold = 6.
grid_score_d60_threshold = 0.85
grid_score_d90_threshold = 1.5
sweep_ids = [
    'zqrq9ri3',  # DoG+Global+CE, various architectures
]

runs_configs_df = download_wandb_project_runs_configs(
    wandb_project_path='mec-hpc-investigations',
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    finished_only=True)

# Keep only networks that achieved low position decoding error.
runs_configs_df = runs_configs_df[
    runs_configs_df['pos_decoding_err'] < low_pos_decoding_err_threshold]

joblib_files_data_by_run_id_dict = load_runs_joblib_files(
    run_ids=list(runs_configs_df['run_id'].unique()))

# runs_histories_df = download_wandb_project_runs_histories(
#     wandb_project_path='mec-hpc-investigations',
#     data_dir=data_dir,
#     sweep_ids=sweep_ids)

runs_augmented_histories_df = runs_configs_df[[
    'run_id', 'optimizer']].merge(
    runs_histories_df,
    on='run_id',
    how='left')

plot_max_grid_score_vs_num_grad_steps_by_optimizer(
    runs_augmented_histories_df=runs_augmented_histories_df,
    plot_dir=results_dir,
    grid_score_d60_threshold=grid_score_d60_threshold,
    grid_score_d90_threshold=grid_score_d90_threshold,
)

plot_loss_vs_num_grad_steps_by_optimizer(
    runs_augmented_histories_df=runs_augmented_histories_df,
    plot_dir=results_dir)

plot_pos_decoding_error_vs_num_grad_steps_by_optimizer(
    runs_augmented_histories_df=runs_augmented_histories_df,
    plot_dir=results_dir)


print('Finished!')
