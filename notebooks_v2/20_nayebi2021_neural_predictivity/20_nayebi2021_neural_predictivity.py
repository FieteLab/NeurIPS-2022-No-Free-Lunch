import os
import pandas as pd
import shutil

from mec_hpc_investigations.models.analyze import *
from mec_hpc_investigations.models.plot import *

# Declare variables.
notebook_dir = 'notebooks_v2/20_nayebi2021_neural_predictivity'
data_dir = os.path.join(notebook_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
results_dir = os.path.join(notebook_dir, 'results')
if os.path.exists(results_dir) and os.path.isdir(results_dir):
    shutil.rmtree(results_dir)
os.makedirs(results_dir, exist_ok=True)

neural_predictivity_df = pd.read_csv(
    os.path.join(data_dir, 'nayebi2021_neural_predictivity.csv'),
    index_col=False)

# 4 columns: Architecture, Nonlinearity, Score Type, Score
# neural_predictivity_df = pd.melt(
#     neural_predictivity_df,
#     id_vars=['Architecture', 'Nonlinearity'],
#     var_name='Score Type',
#     value_name='Score'
# )


sweep_ids = [
    'can8n6vd',  # DoG sweeping architecture
]

runs_configs_df = download_wandb_project_runs_configs(
    wandb_project_path='mec-hpc-investigations',
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    finished_only=True,
    refresh=False)

joblib_files_data_by_run_id_dict = load_runs_joblib_files(
    run_ids=list(runs_configs_df['run_id'].unique()))

rate_maps_participation_ratio_by_run_id_df = compute_rate_maps_participation_ratio_from_joblib_files_data(
    joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict,
    data_dir=data_dir)

runs_configs_df = runs_configs_df.merge(
    rate_maps_participation_ratio_by_run_id_df,
    on=['run_id'],
    how='left')

rate_map_rank_by_run_id_df = compute_rate_maps_rank_from_joblib_files_data(
    joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict,
    data_dir=data_dir)

runs_configs_df = runs_configs_df.merge(
    rate_map_rank_by_run_id_df,
    on=['run_id'],
    how='left')


trained_neural_predictivity_and_ID_df = runs_configs_df[[
    'run_id', 'rnn_type', 'activation', 'participation_ratio', 'rate_maps_participation_ratio',
    'rate_maps_rank', 'two_NN', 'method_of_moments_ID']].merge(
    neural_predictivity_df[['rnn_type', 'activation', 'Trained']],
    on=['rnn_type', 'activation'],
    how='left')

# trained_neural_predictivity_and_ID_df = pd.melt(
#     trained_neural_predictivity_and_ID_df,
#     id_vars=['run_id', 'rnn_type', '3activation', 'Trained'],
#     var_name='Intrinsic Dim Measure',
#     value_name='Intrinsic Dim'
# )

trained_neural_predictivity_and_ID_df.rename(
    columns={'rnn_type': 'Architecture',
             'activation': 'Activation'},
    inplace=True)

plot_neural_predictivity_vs_activity_participation_ratio_by_architecture_and_activation(
    trained_neural_predictivity_and_ID_df=trained_neural_predictivity_and_ID_df,
    plot_dir=results_dir)

plot_neural_predictivity_vs_rate_maps_participation_ratio_by_architecture_and_activation(
    trained_neural_predictivity_and_ID_df=trained_neural_predictivity_and_ID_df,
    plot_dir=results_dir)

plot_neural_predictivity_vs_rate_maps_participation_ratio_by_architecture_and_activation_custom(
    trained_neural_predictivity_and_ID_df=trained_neural_predictivity_and_ID_df,
    plot_dir=results_dir)

plot_neural_predictivity_vs_rate_maps_rank_by_architecture_and_activation(
    trained_neural_predictivity_and_ID_df=trained_neural_predictivity_and_ID_df,
    plot_dir=results_dir)
