import argparse
import numpy as np
import os
import tensorflow as tf
import wandb

from mec_hpc_investigations.models.trainer import Trainer
from mec_hpc_investigations.models.utils import configure_options, configure_model


default_config = {
    'activation': 'sigmoid',
    'batch_size': 200,
    'bin_side_in_m': 0.05,
    'box_height_in_m': 2.2,
    'box_width_in_m': 2.2,
    'initializer': 'glorot_uniform',
    'is_periodic': False,
    'learning_rate': 1e-4,
    'n_epochs': 10,
    'n_grad_steps_per_epoch': 10,
    'n_recurrent_units_to_sample': 16,
    'n_place_fields_per_cell': 1,
    'Np': 512,
    'Ng': 4096,
    'optimizer': 'sgd',
    'place_field_function': 'gaussian',
    'place_field_normalization': 'normal',
    'place_cell_rf': 0.12,
    'readout_dropout': 0.,
    'recurrent_dropout': 0.,
    'rnn_type': 'rnn',
    'seed': 0,
    'sequence_length': 20,
    'surround_scale': 2,
    'weight_decay': 1e-4,
}

wandb.init(project='mec-hpc-investigations',
           config=default_config)
wandb_config = wandb.config

# If GPUs available, select which to train on
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Set seeds.
np.random.seed(seed=wandb_config.seed)
tf.random.set_seed(seed=wandb_config.seed)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

options = configure_options(save_dir='results',
                            run_ID=wandb.run.id,
                            **wandb_config)

model = configure_model(options=options)
trainer = Trainer(options=options,
                  model=model)
trainer.train(save=False,
              log_and_plot_grid_scores=True)

print('Finished training.')
