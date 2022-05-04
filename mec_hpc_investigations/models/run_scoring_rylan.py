import joblib
import numpy as np
import os

from mec_hpc_investigations.models.trainer import Trainer
from mec_hpc_investigations.models.utils import configure_options, configure_model


results_dir = 'results'
wandb_run_id = '0dxm9nbz'
run_dir = os.path.join(results_dir, wandb_run_id)
ckpts_dir_path = os.path.join(run_dir, 'ckpts')
options_path = os.path.join(ckpts_dir_path, 'options.joblib')
place_cells_path = os.path.join(ckpts_dir_path, 'place_cells.joblib')

# Load options, create model, then load place cells and overwrite model's.
options = joblib.load(options_path)

# Specify the length of trajectories for evaluation.
options.sequence_length = 35

# Create model.
model = configure_model(options=options)

# During training, we may have used a particular version of the place cell class.
# However, for evaluation, Rylan may have changed the position decoder while
# trying to debug position decoding from multiple fields per place cell. Because
# the position decoder is part of the PlaceCell class, we instead just copy over
# the relevant data members.
training_place_cells = joblib.load(place_cells_path)
model.place_cells.us = training_place_cells.us
model.place_cells.place_cell_rf = training_place_cells.place_cell_rf
model.place_cells.surround_scale = training_place_cells.surround_scale
model.place_cells.fields_to_delete = training_place_cells.fields_to_delete
model.place_cells.fields_to_keep = training_place_cells.fields_to_keep

# Create the trainer, trajectory generator and grid scorer.
trainer = Trainer(options=options,
                  model=model)
trajectory_gen = trainer.trajectory_generator.get_generator()
trainer.create_grid_scorer()

trainer.eval_after_train(
    gen=trajectory_gen,
    run_dir=run_dir)

print('Finished scoring.')
