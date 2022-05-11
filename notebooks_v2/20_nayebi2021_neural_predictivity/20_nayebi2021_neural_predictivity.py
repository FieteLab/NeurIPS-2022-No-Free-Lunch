import os
import pandas as pd

from mec_hpc_investigations.models.analyze import *
from mec_hpc_investigations.models.plot import *

# Declare variables.
notebook_dir = 'notebooks_v2/20_nayebi2021_neural_predictivity'
data_dir = os.path.join(notebook_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
results_dir = os.path.join(notebook_dir, 'results')
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
    refresh=True)

joblib_files_data_by_run_id_dict = load_runs_joblib_files(
    run_ids=list(runs_configs_df['run_id'].unique()))

overwrite_runs_configs_df_values_with_joblib_data(
    runs_configs_df=runs_configs_df,
    joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict)

