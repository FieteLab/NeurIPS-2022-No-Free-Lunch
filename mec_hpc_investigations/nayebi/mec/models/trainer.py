# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from mec_hpc_investigations.models.visualize import save_ratemaps
from mec_hpc_investigations.models.place_cells import PlaceCells
from mec_hpc_investigations.models.head_direction_cells import HeadDirectionCells
from mec_hpc_investigations.models.trajectory_generator import TrajectoryGenerator
from tqdm import tqdm


class Trainer(object):
    def __init__(self, options, model):
        self.options = options
        self.model = model
        self.hdc = hasattr(self.options, "Nhdc")
        if self.hdc:
            self.trajectory_generator = TrajectoryGenerator(options=self.options,
                                                            place_cells=PlaceCells(self.options),
                                                            head_direction_cells=HeadDirectionCells(self.options))
        else:
            self.trajectory_generator = TrajectoryGenerator(options=self.options,
                                                            place_cells=PlaceCells(self.options))

        lr = self.options.learning_rate
        optimizer_class = self.options.optimizer_class
        optim_kwargs = {}
        # setting clipvalue to None returns error, so only setting it conditionally
        if self.options.clipvalue is not None:
            optim_kwargs["clipvalue"] = self.options.clipvalue
        if optimizer_class.lower() == "adam":
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, **optim_kwargs)
        elif optimizer_class.lower() == "rmsprop":
            # setting momentum to 0.9 to match banino
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr, momentum=0.9, **optim_kwargs)
        else:
            raise ValueError

        self.loss = []
        self.err = []

        # Set up checkpoints
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=self.optimizer, net=model)
        self.ckpt_dir = os.path.join(options.save_dir, options.run_ID, "ckpts")
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_dir, max_to_keep=500)
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print("Restored trained model from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing new model from scratch.")


    def train_step(self, inputs, cell_outputs, pos):
        '''
        Train on one batch of trajectories.

        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            cell_outputs: By default, the ground truth place cell activations with shape
                [batch_size, sequence_length, Np].
                Otherwise dictionary of place cell and head direction cell activations.
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        '''
        with tf.GradientTape() as tape:
            loss, err = self.model.compute_loss(inputs, cell_outputs, pos)

        grads = tape.gradient(loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


        return loss, err


    def train(self, n_epochs=10, n_steps=100, save=True):
        '''
        Train model on simulated trajectories.

        Args:
            n_epochs: Number of training epochs
            n_steps: Number of batches of trajectories per epoch
            save: If true, save a checkpoint after each epoch.
        '''

        # Construct generator
        gen = self.trajectory_generator.get_generator()

        # Save at beginning of training
        if save:
            self.ckpt_manager.save()
            np.save(os.path.join(self.ckpt_dir, "options.npy"), vars(self.options))

        for epoch in tqdm(range(n_epochs)):
            t = tqdm(range(n_steps), leave=False)
            for _ in t:
                inputs, cell_outputs, pos = next(gen)
                loss, err = self.train_step(inputs, cell_outputs, pos)
                self.loss.append(loss)
                self.err.append(err)

                #Log error rate
                t.set_description(f"Error = {100*err} cm")

                self.ckpt.step.assign_add(1)

            if save:
                # Save checkpoint
                self.ckpt_manager.save()
                tot_step = self.ckpt.step.numpy()

                # Save a picture of rate maps
                save_ratemaps(self.model, self.options, step=tot_step)


    def load_ckpt(self, idx):
        ''' Restore model from earlier checkpoint. '''
        self.ckpt.restore(self.ckpt_manager.checkpoints[idx])
