import joblib
import numpy as np
import os
import sys
import tensorflow as tf

# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from mec_hpc_investigations.models.trainer import Trainer
from mec_hpc_investigations.models.utils import configure_options, configure_model


results_dir = 'results'
wandb_run_id = sys.argv[1]  # '0dxm9nbz'
# wandb_run_id = 'gaqt4cge'
print(f'W&B Run ID: {wandb_run_id}')
run_dir = os.path.join(results_dir, wandb_run_id)
ckpts_dir_path = os.path.join(run_dir, 'ckpts')
options_path = os.path.join(ckpts_dir_path, 'options.joblib')
place_cells_path = os.path.join(ckpts_dir_path, 'place_cells.joblib')

# Load options, create model, then load place cells and overwrite model's.
options = joblib.load(options_path)
print('Loaded options')

# Specify the length of trajectories for evaluation.
options.sequence_length = 50
# options.batch_size = 30

# Create model.
model = configure_model(options=options)
print('Loaded model')


# During training, we may have used a particular version of the place cell class.
# However, for evaluation, Rylan may have changed the position decoder while
# trying to debug position decoding from multiple fields per place cell. Because
# the position decoder is part of the PlaceCell class, we instead just copy over
# the relevant data members.
if options.place_field_values != 'cartesian':
    training_place_cells = joblib.load(place_cells_path)
    model.place_cells.us = training_place_cells.us
    model.place_cells.place_cell_rf = training_place_cells.place_cell_rf
    model.place_cells.surround_scale = training_place_cells.surround_scale
    model.place_cells.fields_to_delete = training_place_cells.fields_to_delete
    model.place_cells.fields_to_keep = training_place_cells.fields_to_keep
    print('Loaded place cells.')

# Create the trainer, trajectory generator and grid scorer.
trainer = Trainer(options=options,
                  model=model,
                  split='eval')
print('Created trainer.')
trajectory_gen = trainer.trajectory_generator.get_generator()
print('Created trajectory generator.')
trainer.create_grid_scorer()
print('Created grid scorer.')

trainer.eval_after_train(
    gen=trajectory_gen,
    run_dir=run_dir,
    # refresh=True,
    refresh=False,
)

print('Finished scoring.')
