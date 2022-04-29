import joblib
import numpy as np
import os

from mec_hpc_investigations.models.trainer import Trainer
from mec_hpc_investigations.models.utils import configure_options, configure_model


results_dir = 'results'
run_id = 'wtylubz8'
run_id_dir = os.path.join(results_dir, run_id)
ckpts_dir_path = os.path.join(run_id_dir, 'ckpts')
options_path = os.path.join(ckpts_dir_path, 'options.joblib')
place_cells_path = os.path.join(ckpts_dir_path, 'place_cells.joblib')

# Load options, create model, then load place cells and overwrite model's.
options = joblib.load(options_path)
model = configure_model(options=options)
place_cells = joblib.load(place_cells_path)
model.place_cells = place_cells
trainer = Trainer(options=options,
                  model=model)
print('Finished scoring.')
