import argparse
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.profiler import profiler_v2 as profiler
import wandb

from mec_hpc_investigations.models.trainer import Trainer
from mec_hpc_investigations.models.utils import configure_options, configure_model


# Position config.
default_config = {
    # 'activation': 'tanh',
    'activation': 'relu',
    'batch_size': 13,
    'bin_side_in_m': 0.05,
    'box_height_in_m': 2.2,
    'box_width_in_m': 2.2,
    'initializer': 'glorot_uniform',
    'is_periodic': False,
    'learning_rate': 1e-4,
    'n_epochs': 3,
    'n_grad_steps_per_epoch': 5,
    'n_recurrent_units_to_sample': 16,
    'n_place_fields_per_cell': 1,
    # 'n_place_fields_per_cell': 'Poisson ( 1.5 )',
    'Np': 32,
    'Ng': 256,
    'optimizer': 'adam',
    'place_field_loss': 'crossentropy',
    # 'place_field_values': 'gaussian',
    'place_field_values': 'difference_of_gaussians',
    'place_field_normalization': 'global',
    'place_cell_rf': 0.12,
    # 'place_cell_rf': 'Uniform( 0.09 , 0.15 )',  # WARNING: Spaces needed
    'readout_dropout': 0.,
    'recurrent_dropout': 0.,
    # 'rnn_type': 'UGRNN',
    'rnn_type': 'RNN',
    'seed': 0,
    'sequence_length': 20,
    'surround_scale': 2.,
    # 'surround_scale': 'Uniform( 1.9 , 2.1 )',
    'weight_decay': 1e-4,
}

wandb.init(project='mec-hpc-investigations',
           config=default_config)
wandb_config = wandb.config


# Check if config is valid.


# If GPUs available, select which to train on
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Set seeds.
np.random.seed(seed=wandb_config.seed)
tf.random.set_seed(seed=wandb_config.seed)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

options = configure_options(save_dir='results',
                            run_ID=wandb.run.id,
                            **wandb_config)

# profiler.warmup()
# profiler.start(logdir='logdir')
model = configure_model(options=options)
trainer = Trainer(options=options,
                  model=model)
trainer.train(save=False,
              log_and_plot_grid_scores=False)
# profiler.stop()
print('Finished training.')
