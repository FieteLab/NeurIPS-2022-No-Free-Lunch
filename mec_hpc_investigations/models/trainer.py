# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from tqdm import tqdm
import wandb

from mec_hpc_investigations.models.visualize import save_ratemaps
from mec_hpc_investigations.models.helper_classes import Options, PlaceCells
from mec_hpc_investigations.models.scores import GridScorer
from mec_hpc_investigations.models.trajectory_generator import TrajectoryGenerator
from mec_hpc_investigations.models.utils import compute_ratemaps


class Trainer(object):

    def __init__(self,
                 options: Options,
                 model):
        self.options = options
        self.model = model
        self.trajectory_generator = TrajectoryGenerator(self.options, PlaceCells(self.options))
        lr = self.options.learning_rate

        if options.optimizer == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif options.optimizer == 'adagrad':
            self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
        elif options.optimizer == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        elif options.optimizer == 'rmsprop':
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        else:
            raise NotImplementedError

        self.loss = []
        self.err = []

        self.scorer = None

        # Set up checkpoints
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=self.optimizer, net=model)
        self.ckpt_dir = os.path.join(options.save_dir, options.run_ID, "ckpts")
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_dir, max_to_keep=500)
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print("Restored trained model from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing new model from scratch.")

    def train_step(self, inputs, pc_outputs, pos):
        '''
        Train on one batch of trajectories.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].
        Returns:
            loss: Avg. loss for this training batch.
            pos_decoding_err: Avg. decoded position error in cm.
        '''
        with tf.GradientTape() as tape:
            loss, pos_decoding_err = self.model.compute_loss(inputs, pc_outputs, pos)

        grads = tape.gradient(loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss, pos_decoding_err

    def train(self,
              save: bool = True,
              log_and_plot_grid_scores: bool = False):
        '''
        Train model on simulated trajectories.
        Args:
            n_epochs: Number of training epochs
            n_grad_steps_per_epoch: Number of batches of trajectories per epoch
            save: If true, save a checkpoint after each epoch.
        '''

        # Construct generator
        gen = self.trajectory_generator.get_generator()

        # Save at beginning of training
        if save:
            self.ckpt_manager.save()
            np.save(os.path.join(self.ckpt_dir, "options.npy"), vars(self.options))

        if log_and_plot_grid_scores:
            # TODO: these values might need to be recalculated
            # 0.2 is fractional amount away from wall
            starts = [0.2] * 10
            ends = np.linspace(0.4, 1.0, num=10)
            coords_range = ((-1.1, 1.1), (-1.1, 1.1))
            nbins = int((coords_range[0][1] - coords_range[0][0]) / self.options.bin_side_in_m)
            masks_parameters = zip(starts, ends.tolist())
            self.scorer = GridScorer(nbins=nbins,
                                     mask_parameters=masks_parameters,
                                     coords_range=coords_range)

        for epoch_idx in tqdm(range(self.options.n_epochs)):
            t = tqdm(range(self.options.n_grad_steps_per_epoch), leave=False)
            for _ in t:
                inputs, pc_outputs, pos = next(gen)
                loss, pos_decoding_err = self.train_step(inputs, pc_outputs, pos)
                self.loss.append(loss)
                self.err.append(pos_decoding_err)

                # Log error rate
                t.set_description(f"Error = {100 * pos_decoding_err} cm")

                self.ckpt.step.assign_add(1)

            if save:
                # Save checkpoint
                self.ckpt_manager.save()
                tot_step = self.ckpt.step.numpy()

                # Save a picture of rate maps
                save_ratemaps(self.model, self.options, step=tot_step)

            wandb.log({
                'loss': loss,
                'pos_decoding_err': 100 * pos_decoding_err,
            }, step=epoch_idx)

            if log_and_plot_grid_scores:
                self.log_and_plot_grid_scores(pos=pos, inputs=inputs,
                                              epoch_idx=epoch_idx)

    def load_ckpt(self, idx):
        ''' Restore model from earlier checkpoint. '''
        self.ckpt.restore(self.ckpt_manager.checkpoints[idx])

    def log_and_plot_grid_scores(self,
                                 pos: tf.Tensor,
                                 inputs: tf.Tensor,
                                 epoch_idx: int):

        xs = tf.reshape(
            pos[:, :, 0],
            shape=[self.options.batch_size * self.options.sequence_length])
        ys = tf.reshape(
            pos[:, :, 1],
            shape=[self.options.batch_size * self.options.sequence_length])

        activations = tf.reshape(
            tf.stop_gradient(self.model.g(inputs)),
            shape=[self.options.batch_size * self.options.sequence_length, self.options.Ng]
        )

        n_samples = self.options.n_recurrent_units_to_sample
        print('n_samples: ', n_samples)
        score_60_by_neuron = np.zeros(n_samples)
        score_90_by_neuron = np.zeros(n_samples)

        best_score_60 = -np.inf
        best_rate_map_60 = None
        best_score_90 = -np.inf
        best_rate_map_90 = None

        vmin = np.min(activations)
        vmax = np.max(activations)

        neuron_indices = np.random.choice(self.options.Ng, replace=False,
                                          size=n_samples)

        rate_maps = np.zeros(shape=(n_samples, self.scorer._nbins, self.scorer._nbins))

        for storage_idx, neuron_idx in enumerate(neuron_indices):

            # Would recommend switching to visualize.py's `compute_ratemaps`
            # Then util.py's `get_model_activations`
            # then utils.py's get_model_gridscores
            #    model_resp is the rate map for a single layer
            rate_map = self.scorer.calculate_ratemap(
                xs=xs,
                ys=ys,
                activations=activations[:, neuron_idx],
            )
            scores = self.scorer.get_scores(rate_map=rate_map)
            score_60 = scores[0]
            if score_60 > best_score_60:
                best_score_60 = score_60
                best_rate_map_60 = rate_map

            score_90 = scores[1]
            if score_90 > best_score_90:
                best_score_90 = score_90
                best_rate_map_90 = rate_map

            score_60_by_neuron[storage_idx] = score_60
            score_90_by_neuron[storage_idx] = score_90
            rate_maps[storage_idx] = rate_map

        n_rows = n_cols = int(np.sqrt(n_samples))

        fig, axes = plt.subplots(
            n_rows,  # rows
            n_cols,  # columns
            figsize=(2 * n_rows, 2 * n_cols),
            sharey=True,
            sharex=True,
            gridspec_kw={'width_ratios': [1] * n_cols})

        storage_idx_sorted_by_score_60 = np.argsort(score_60_by_neuron)[::-1]
        for count_idx, storage_idx in enumerate(storage_idx_sorted_by_score_60):

            row, col = count_idx // n_cols, count_idx % n_cols
            ax = axes[row, col]

            sns.heatmap(
                data=rate_maps[storage_idx],
                vmin=vmin,
                vmax=vmax,
                ax=ax,
                cbar=False,
                cmap='rocket_r',
                yticklabels=False,
                xticklabels=False)

            ax.set_title(f'60={np.round(score_60_by_neuron[storage_idx], 2)}:'
                         f'90={np.round(score_90_by_neuron[storage_idx], 2)}')

            # seaborn heatmap frustratingly flips the y axis for some reason
            ax.invert_yaxis()

        best_rate_map_60[np.isnan(best_rate_map_60)] = 0.
        best_rate_map_90[np.isnan(best_rate_map_90)] = 0.

        wandb.log({
            f'rate_maps': wandb.Image(fig),
            f'max_grid_score_d=60_n={n_samples}': np.max(score_60_by_neuron),
            f'grid_score_histogram_d=60_n={n_samples}': wandb.Histogram(score_60_by_neuron),
            f'best_rate_map_d=60_n={n_samples}': best_rate_map_60,
            # 'best_rate_map_d=60_n=128': wandb.plots.HeatMap(matrix_values=best_rate_map_60,
            #                                                 x_labels=np.arange(best_rate_map_60.shape[1]),
            #                                                 y_labels=np.arange(best_rate_map_60.shape[0])),
            f'max_grid_score_d=90_n={n_samples}': np.max(score_90_by_neuron),
            f'grid_score_histogram_d=90_n={n_samples}': wandb.Histogram(score_90_by_neuron),
            f'best_rate_map_d=90_n={n_samples}': best_rate_map_90,
            # 'best_rate_map_d=90': wandb.plots.HeatMap(matrix_values=best_rate_map_90,
            #                                           x_labels=np.arange(best_rate_map_90.shape[1]),
            #                                           y_labels=np.arange(best_rate_map_90.shape[0])),
        }, step=epoch_idx)

        # plt.show()
        plt.close(fig=fig)

