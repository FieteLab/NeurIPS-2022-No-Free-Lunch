# -*- coding: utf-8 -*-
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import skdim
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm
import wandb

from mec_hpc_investigations.models.visualize import save_ratemaps
from mec_hpc_investigations.models.helper_classes import Options, PlaceCells
from mec_hpc_investigations.models.scores import GridScorer
from mec_hpc_investigations.models.trajectory_generator import TrajectoryGenerator


# from mec_hpc_investigations.models.utils import compute_and_log_rate_maps


class Trainer(object):

    def __init__(self,
                 options: Options,
                 model):
        self.options = options
        self.model = model

        # Original
        # Removed because we shouldn't duplicate PlaceCells
        # self.trajectory_generator = TrajectoryGenerator(self.options, PlaceCells(self.options))
        self.trajectory_generator = TrajectoryGenerator(
            options=self.options,
            place_cells=model.place_cells)
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
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_dir, max_to_keep=500)
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print("Restored trained model from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing new model from scratch.")

    def create_grid_scorer(self):
        # TODO: these values might need to be recalculated
        # 0.2 is fractional amount away from wall
        starts = [0.2] * 10
        ends = np.linspace(0.4, 1.0, num=10)

        # TODO: Are these height and widths in the correct order?
        # coords_range = ((-1.1, 1.1), (-1.1, 1.1))
        coords_range = (
            (-self.options.box_width_in_m / 2., self.options.box_width_in_m / 2.),
            (-self.options.box_height_in_m / 2., self.options.box_height_in_m / 2.))
        nbins = int((coords_range[0][1] - coords_range[0][0]) / self.options.bin_side_in_m)
        masks_parameters = zip(starts, ends.tolist())
        self.scorer = GridScorer(nbins=nbins,
                                 mask_parameters=masks_parameters,
                                 coords_range=coords_range)

    def eval_step(self, inputs, pc_outputs, pos):
        '''
        Eval on one batch of trajectories.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].
        Returns:
            loss: Avg. loss for this training batch.
            pos_decoding_err: Avg. decoded position error in cm.
        '''

        loss, pos_decoding_err = self.model.compute_loss(inputs, pc_outputs, pos)

        return loss.numpy(), pos_decoding_err.numpy()

    def eval_during_train(self,
                          gen: TrajectoryGenerator,
                          epoch_idx: int,
                          n_samples: int = 10,
                          save: bool = False,
                          # log_and_plot_grid_scores: bool = True,
                          ):

        inputs, pc_outputs, pos = next(gen)
        loss, pos_decoding_err = self.eval_step(inputs, pc_outputs, pos)
        wandb_vals_to_log = {
            'loss': loss,
            'pos_decoding_err': 100 * pos_decoding_err,
        }

        wandb_vals_to_log.update(self.compute_intrinsic_dimensionalities(
            inputs=inputs))

        wandb.log(wandb_vals_to_log, step=epoch_idx + 1)

        self.model.log_weight_norms(epoch_idx=epoch_idx)

        if save:
            # Save checkpoint
            self.ckpt_manager.save()
            # tot_step = self.ckpt.step.numpy()

            # Save a picture of rate maps
            # save_ratemaps(self.model, self.options, step=tot_step)

        # if log_and_plot_grid_scores:
        #     self.log_and_plot_all(pos=pos,
        #                           inputs=inputs,
        #                           epoch_idx=epoch_idx,
        #                           n_samples=n_samples)

    def eval_after_train(self,
                         gen: TrajectoryGenerator,
                         run_dir: str,
                         n_samples: int = None,
                         ):

        if n_samples is None:
            n_samples = self.options.Ng

        inputs, pc_outputs, pos = next(gen)
        loss, pos_decoding_err = self.eval_step(inputs, pc_outputs, pos)
        intrinsic_dimensionalities = self.compute_intrinsic_dimensionalities(
            inputs=inputs)
        results = self.log_and_plot_all(pos=pos,
                                        inputs=inputs,
                                        epoch_idx=None,
                                        n_samples=n_samples,
                                        log_to_wandb=False,
                                        run_dir=run_dir)
        print(10)

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

            # Save options
            np.save(os.path.join(self.ckpt_dir, "options.npy"), vars(self.options))
            joblib.dump(self.options, os.path.join(self.ckpt_dir, 'options.joblib'))

            # Save model place cells.
            joblib.dump(self.model.place_cells, os.path.join(self.ckpt_dir, 'place_cells.joblib'))

        if log_and_plot_grid_scores:
            self.create_grid_scorer()

        assert self.options.n_epochs > 0
        for epoch_idx in tqdm(range(self.options.n_epochs)):

            # t = tqdm(range(self.options.n_grad_steps_per_epoch), leave=False)
            for _ in range(self.options.n_grad_steps_per_epoch):
                inputs, pc_outputs, pos = next(gen)
                loss, pos_decoding_err = self.train_step(inputs, pc_outputs, pos)
                self.loss.append(loss)
                self.err.append(pos_decoding_err)

                # Log error rate
                # t.set_description(f"Error = {100 * pos_decoding_err} cm")

                self.ckpt.step.assign_add(1)

            self.eval_during_train(
                gen=gen,
                epoch_idx=epoch_idx,
                save=False,
                log_and_plot_grid_scores=False,
            )

        self.eval_during_train(
            gen=gen,
            epoch_idx=epoch_idx,
            save=True,
            log_and_plot_grid_scores=log_and_plot_grid_scores,
        )

    def load_ckpt(self, idx):
        ''' Restore model from earlier checkpoint. '''
        self.ckpt.restore(self.ckpt_manager.checkpoints[idx])

    def log_and_plot_all(self,
                         pos: tf.Tensor,
                         inputs: tf.Tensor,
                         epoch_idx: int,
                         n_samples: int,
                         log_to_wandb: bool = True,
                         run_dir: str = None):

        if log_to_wandb:
            num_grad_steps_taken = epoch_idx * self.options.n_grad_steps_per_epoch
            num_trajectories_trained_on = num_grad_steps_taken * self.options.batch_size

            wandb.log({
                'num_grad_steps': num_grad_steps_taken,
                'num_trajectories_trained_on': num_trajectories_trained_on,
            }, step=epoch_idx + 1)

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

        rate_maps, score_60_by_neuron, score_90_by_neuron = self.compute_and_log_rate_maps(
            xs=xs,
            ys=ys,
            activations=activations,
            epoch_idx=epoch_idx,
            n_samples=n_samples,
            log_to_wandb=log_to_wandb,
            run_dir=run_dir)

        period_per_cell, period_err_per_cell, orientations_per_cell = self.compute_and_log_grid_cell_periodicity_and_orientation(
            rate_maps=rate_maps,
            score_60_by_neuron=score_60_by_neuron,
            score_90_by_neuron=score_90_by_neuron,
            epoch_idx=epoch_idx,
            log_to_wandb=log_to_wandb,
            run_dir=run_dir,
        )

        self.plot_and_log_ratemaps(
            rate_maps=rate_maps,
            score_60_by_neuron=score_60_by_neuron,
            score_90_by_neuron=score_90_by_neuron,
            epoch_idx=epoch_idx,
            log_to_wandb=log_to_wandb,
            run_dir=run_dir)

        results = dict(
            rate_maps=rate_maps,
            score_60_by_neuron=score_60_by_neuron,
            score_90_by_neuron=score_90_by_neuron,
            period_per_cell=period_per_cell,
            period_err_per_cell=period_err_per_cell,
            orientations_per_cell=orientations_per_cell,
        )

        return results

    def compute_intrinsic_dimensionalities(self,
                                           inputs):
        # activations has shape: (batch_size * sequence_length, Ng)
        activations = tf.reshape(
            tf.stop_gradient(self.model.g(inputs)),
            shape=[self.options.batch_size * self.options.sequence_length, self.options.Ng]
        )

        # TODO: add ID measures to rate maps (spatial vs neuron)

        # In this function, ID stands for Intrinsic Dimensionality.
        two_NN_ID = skdim.id.TwoNN().fit_transform(X=activations)

        method_of_moments_ID = skdim.id.MOM().fit_transform(X=activations)

        # Use skdim implementation for trustworthiness.
        participation_ratio_ID = skdim.id.lPCA(ver='participation_ratio').fit_transform(
            X=activations)

        intrinsic_dimensionalities = dict(
            participation_ratio=participation_ratio_ID,
            two_NN=two_NN_ID,
            method_of_moments_ID=method_of_moments_ID,
        )

        return intrinsic_dimensionalities

    def compute_and_log_grid_cell_periodicity_and_orientation(self,
                                                              rate_maps: np.ndarray,
                                                              score_60_by_neuron: np.ndarray,
                                                              score_90_by_neuron: np.ndarray,
                                                              epoch_idx: int,
                                                              run_dir: str,
                                                              threshold: float = 0.9,
                                                              log_to_wandb: bool = True):

        period_results_joblib_path = os.path.join(run_dir, 'period_results_path.joblib')
        if not os.path.isfile(period_results_joblib_path):

            likely_grid_cell_indices = score_60_by_neuron > threshold
            if np.sum(likely_grid_cell_indices) == 0:
                return

            period_per_cell, period_err_per_cell, orientations_per_cell = [], [], []
            for rate_map in rate_maps[likely_grid_cell_indices]:
                rate_map_copy = np.copy(rate_map)
                # NOTE: This is new as of 2022/04/19.
                rate_map_copy[np.isnan(rate_map_copy)] = 0.
                period, period_err, orientations = self.scorer.calculate_grid_cell_periodicity_and_orientation(
                    rate_map=rate_map_copy)
                period_per_cell.append(period)
                period_err_per_cell.append(period_err)
                orientations_per_cell.append(orientations.tolist())

            if log_to_wandb:
                wandb.log({
                    f'period_per_cell_threshold={threshold}': period_per_cell,
                    f'period_err_per_cell_threshold={threshold}': period_err_per_cell,
                    f'orientations_per_cell_threshold={threshold}': orientations_per_cell,
                }, step=epoch_idx + 1)

            joblib.dump(
                {'period_per_cell': period_per_cell,
                 'period_err_per_cell': period_err_per_cell,
                 'orientations_per_cell': orientations_per_cell,
                 'threshold': threshold},
                filename=period_results_joblib_path
            )
        else:
            previously_generated_results = joblib.load(period_results_joblib_path)
            previous_threshold = previously_generated_results['threshold']
            if previous_threshold != threshold:
                raise ValueError(f'Previous threshold ({previous_threshold}) does not equal desired threshold ({threshold}).')
            period_per_cell = previously_generated_results['period_per_cell']
            period_err_per_cell = previously_generated_results['period_err_per_cell']
            orientations_per_cell = previously_generated_results['orientations_per_cell']

        return period_per_cell, period_err_per_cell, orientations_per_cell

    def compute_and_log_rate_maps(self,
                                  xs,
                                  ys,
                                  activations,
                                  epoch_idx: int,
                                  n_samples: int,
                                  run_dir: str,
                                  log_to_wandb: bool = True):

        rate_maps_and_scores_joblib_path = os.path.join(run_dir, 'rate_maps_and_scores.joblib')
        if not os.path.isfile(rate_maps_and_scores_joblib_path):
            # Create scores and ratemaps, then save to dis.
            score_60_by_neuron = np.zeros(n_samples)
            score_90_by_neuron = np.zeros(n_samples)

            best_score_60 = -np.inf
            best_rate_map_60 = None
            best_score_90 = -np.inf
            best_rate_map_90 = None

            neuron_indices = np.random.choice(self.options.Ng, replace=False,
                                              size=n_samples)

            rate_maps = np.zeros(shape=(n_samples, self.scorer._nbins, self.scorer._nbins))

            for storage_idx, neuron_idx in enumerate(neuron_indices):

                # Would recommend switching to visualize.py's `compute_and_log_rate_maps`
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

            # These NaNs need to be zeroed out if logging as heatmaps to W&B,
            # otherwise W&B throws an error.
            best_rate_map_60[np.isnan(best_rate_map_60)] = 0.
            best_rate_map_90[np.isnan(best_rate_map_90)] = 0.

            # Sometimes, scores can be NaN if a neuron's ratemap is all 0.
            # Remove these.
            score_60_by_neuron = score_60_by_neuron[~np.isnan(score_60_by_neuron)]
            score_90_by_neuron = score_90_by_neuron[~np.isnan(score_90_by_neuron)]

            # W&B recommends saving
            # wandb.run.id
            if log_to_wandb:
                wandb.log({
                    f'max_grid_score_d=60_n={n_samples}': np.nanmax(score_60_by_neuron),
                    f'grid_score_histogram_d=60_n={n_samples}': wandb.Histogram(score_60_by_neuron),
                    f'best_rate_map_d=60_n={n_samples}': best_rate_map_60,
                    f'max_grid_score_d=90_n={n_samples}': np.nanmax(score_90_by_neuron),
                    f'grid_score_histogram_d=90_n={n_samples}': wandb.Histogram(score_90_by_neuron),
                    f'best_rate_map_d=90_n={n_samples}': best_rate_map_90,
                }, step=epoch_idx + 1)

            joblib.dump(
                {'rate_maps': rate_maps,
                 'score_60_by_neuron': score_60_by_neuron,
                 'score_90_by_neuron': score_90_by_neuron},
                filename=rate_maps_and_scores_joblib_path
            )

        else:
            previously_generated_results = joblib.load(rate_maps_and_scores_joblib_path)
            rate_maps = previously_generated_results['rate_maps']
            score_60_by_neuron = previously_generated_results['score_60_by_neuron']
            score_90_by_neuron = previously_generated_results['score_90_by_neuron']

        return rate_maps, score_60_by_neuron, score_90_by_neuron

    def plot_and_log_ratemaps(self,
                              rate_maps,
                              score_60_by_neuron,
                              score_90_by_neuron,
                              epoch_idx: int,
                              log_to_wandb: bool = True,
                              run_dir: str = None):

        n_samples = rate_maps.shape[0]
        n_rows = n_cols = int(np.ceil(np.sqrt(n_samples)))

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
                vmin=np.nanmin(rate_maps[storage_idx]),
                vmax=np.nanmax(rate_maps[storage_idx]),
                ax=ax,
                cbar=False,
                # cmap='rocket_r',
                cmap='Spectral_r',
                yticklabels=False,
                xticklabels=False)

            ax.set_title(f'60={np.round(score_60_by_neuron[storage_idx], 2)}:'
                         f'90={np.round(score_90_by_neuron[storage_idx], 2)}')

            # Seaborn's heatmap flips the y-axis by default. Flip it back ourselves.
            ax.invert_yaxis()

        if log_to_wandb:
            wandb.log({
                f'rate_maps': wandb.Image(fig),
            }, step=epoch_idx + 1)

        if run_dir is not None:
            plt.savefig(os.path.join(run_dir,
                                     f'ratemaps.png'),
                        bbox_inches='tight',
                        dpi=300)

        # plt.show()
        plt.close(fig=fig)
