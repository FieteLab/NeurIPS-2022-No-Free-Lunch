import joblib
import numpy as np
import os

from mec_hpc_investigations.models.trainer import Trainer
from mec_hpc_investigations.models.utils import configure_options, configure_model


results_dir = 'results'
wandb_run_id = '0dxm9nbz'
run_id_dir = os.path.join(results_dir, wandb_run_id)
ckpts_dir_path = os.path.join(run_id_dir, 'ckpts')
options_path = os.path.join(ckpts_dir_path, 'options.joblib')
place_cells_path = os.path.join(ckpts_dir_path, 'place_cells.joblib')

# Load options, create model, then load place cells and overwrite model's.
options = joblib.load(options_path)

# Overwrite options.sequence_length so trajectory generate creates longer trajectories
options.sequence_length = 40

model = configure_model(options=options)
place_cells = joblib.load(place_cells_path)
model.place_cells = place_cells
trainer = Trainer(options=options,
                  model=model)
trajectory_gen = trainer.trajectory_generator.get_generator()
trainer.create_grid_scorer()

trainer.eval_during_train(
    gen=trajectory_gen,
    epoch_idx=epoch_idx,
    save=True,
    log_and_plot_grid_scores=log_and_plot_grid_scores,
)
print('Finished scoring.')
