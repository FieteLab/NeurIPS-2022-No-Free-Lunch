# -*- coding: utf-8 -*-
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import skdim
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm
from typing import Dict
import wandb

from mec_hpc_investigations.models.visualize import save_ratemaps
from mec_hpc_investigations.models.helper_classes import Options, PlaceCells
from mec_hpc_investigations.models.scores import GridScorer
from mec_hpc_investigations.models.trajectory_generator import TrajectoryGenerator


# from mec_hpc_investigations.models.utils import compute_and_log_rate_maps


class Trainer(object):

    def __init__(self,
                 options: Options,
                 model,
                 split: str = 'train'):

        assert split in {'train', 'eval'}

        self.options = options
        self.model = model
        self.split = split

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

        self.scorers = None

        # Set up checkpoints
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=self.optimizer, net=model)
        self.ckpt_dir = os.path.join(options.save_dir, options.run_ID, "ckpts")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_dir, max_to_keep=500)
        if len(self.ckpt_manager.checkpoints) == 1 and split == 'eval':
            raise ValueError(
                'We currently only want to evaluate a trained model. 1 checkpoint means the model did not finish training.')
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print("Restored trained model from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing new model from scratch.")

    def create_grid_scorer(self):
        # TODO: these values might need to be recalculated
        # 0.2 is fractional amount away from wall
        starts = [0.2] * 10
        # starts = [0.3] * 10
        ends = np.linspace(0.4, 1.0, num=10)

        # coords_range = ((-1.1, 1.1), (-1.1, 1.1))
        coords_range = (
            (-self.options.box_width_in_m / 2., self.options.box_width_in_m / 2.),
            (-self.options.box_height_in_m / 2., self.options.box_height_in_m / 2.))
        # Aran uses 20: https://github.com/neuroailab/NormativeMEC/blob/3d2aa83263732c461b298c3edb82ba05b1a1102e/scripts/bordergridscore_rnn_best_layer.py#L53
        # Ben uses 20: https://github.com/ganguli-lab/grid-pattern-formation/blob/8dcd3907274c5285c7cf756a7f21855f03c198f2/visualize.py#L136
        # nbins = 20
        # Aran said to use 5 cm
        # nbins = int((coords_range[0][1] - coords_range[0][0]) / self.options.bin_side_in_m)
        # masks_parameters = zip(starts, ends.tolist())
        self.scorers = [GridScorer(nbins=nbins,
                                   mask_parameters=zip(starts, ends.tolist()),
                                   coords_range=coords_range)
                        for nbins in [44, 20, 32]]

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

        loss, preds = self.model.compute_loss(inputs, pc_outputs, pos)

        # Compute decoding error
        pred_pos = tf.stop_gradient(self.model.place_cells.get_nearest_cell_pos(preds))
        pos_decoding_err = tf.reduce_mean(tf.sqrt(tf.reduce_sum((tf.cast(pos, dtype=pred_pos.dtype) - pred_pos) ** 2, axis=-1)))

        return loss.numpy(), pos_decoding_err.numpy()

    def eval_during_train(self,
                          gen: TrajectoryGenerator,
                          epoch_idx: int,
                          n_recurr_units_to_analyze: int = 10,
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
        #                           n_recurr_units_to_analyze=n_recurr_units_to_analyze)

    def eval_after_train(self,
                         gen: TrajectoryGenerator,
                         run_dir: str,
                         n_recurr_units_to_analyze: int = None,
                         refresh: bool = False,
                         ):

        # We only use 1024 of networks' recurrent units for the analyses.
        # For most networks, this is 100% of their recurrent units.
        if n_recurr_units_to_analyze is None:
            # n_recurr_units_to_analyze = self.options.Ng
            n_recurr_units_to_analyze = 1024

        inputs, pc_outputs, pos = next(gen)

        loss_pos_and_dims_path = os.path.join(run_dir, 'loss_pos_and_dimensionalities.joblib')
        if refresh or not os.path.isfile(loss_pos_and_dims_path):
            loss, pos_decoding_err = self.eval_step(inputs, pc_outputs, pos)
            pos_decoding_err *= 100  # Convert position decoding error from meters to cm
            values_to_dump = {'loss': loss, 'pos_decoding_err': pos_decoding_err}
            intrinsic_dimensionalities = self.compute_intrinsic_dimensionalities(
                inputs=inputs)
            values_to_dump.update(intrinsic_dimensionalities)
            joblib.dump(values_to_dump, loss_pos_and_dims_path)
            print(values_to_dump)

        self.log_and_plot_all(pos=pos,
                              inputs=inputs,
                              epoch_idx=None,
                              n_recurr_units_to_analyze=n_recurr_units_to_analyze,
                              log_to_wandb=False,
                              run_dir=run_dir,
                              refresh=refresh)

        print('Finished eval after training')

    @tf.function
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
        # By default, the resources held by a GradientTape are released
        # as soon as GradientTape.gradient() method is called.
        with tf.GradientTape() as tape:
            loss, preds = self.model.compute_loss(inputs, pc_outputs, pos)

        tf.debugging.assert_all_finite(loss, 'Loss is not finite')
        grads = tape.gradient(loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss, preds

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
                loss, _ = self.train_step(inputs, pc_outputs, pos)
                self.loss.append(loss)

                # Log error rate
                # t.set_description(f"Error = {100 * pos_decoding_err} cm")

                self.ckpt.step.assign_add(1)

            self.eval_during_train(
                gen=gen,
                epoch_idx=epoch_idx,
                save=False,
                # log_and_plot_grid_scores=False,
            )

        self.eval_during_train(
            gen=gen,
            epoch_idx=epoch_idx,
            save=True,
            # log_and_plot_grid_scores=log_and_plot_grid_scores,
        )

    def load_ckpt(self, idx):
        ''' Restore model from earlier checkpoint. '''
        self.ckpt.restore(self.ckpt_manager.checkpoints[idx])

    def log_and_plot_all(self,
                         pos: tf.Tensor,
                         inputs: tf.Tensor,
                         epoch_idx: int,
                         n_recurr_units_to_analyze: int,
                         log_to_wandb: bool = True,
                         run_dir: str = None,
                         refresh: bool = False):

        if log_to_wandb:
            num_grad_steps_taken = epoch_idx * self.options.n_grad_steps_per_epoch
            num_trajectories_trained_on = num_grad_steps_taken * self.options.batch_size

            wandb.log({
                'num_grad_steps': num_grad_steps_taken,
                'num_trajectories_trained_on': num_trajectories_trained_on,
            }, step=epoch_idx + 1)

        trajectories_and_activations_joblib_path = os.path.join(run_dir, 'trajectories_and_activations.joblib')
        if refresh or not os.path.isfile(trajectories_and_activations_joblib_path):

            print('Generating trajectories and activations.')

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

            joblib.dump(
                {'activations': activations,
                 'xs': xs,
                 'ys': ys},
                filename=trajectories_and_activations_joblib_path)

            print('Saved generated trajectories and activations to disk.')
        else:
            previously_generated_trajectories_and_activations = joblib.load(trajectories_and_activations_joblib_path)
            activations = previously_generated_trajectories_and_activations['activations']
            xs = previously_generated_trajectories_and_activations['xs']
            ys = previously_generated_trajectories_and_activations['ys']
            print('Loaded previously generated trajectories and activations from disk.')

        rate_maps_and_scores_dict = self.compute_and_log_rate_maps(
            xs=xs,
            ys=ys,
            activations=activations,
            epoch_idx=epoch_idx,
            n_recurr_units_to_analyze=n_recurr_units_to_analyze,
            log_to_wandb=log_to_wandb,
            run_dir=run_dir,
            refresh=refresh)

        self.compute_and_log_grid_cell_periodicity_and_orientation(
            rate_maps_and_scores_dict=rate_maps_and_scores_dict,
            epoch_idx=epoch_idx,
            log_to_wandb=log_to_wandb,
            run_dir=run_dir,
            refresh=refresh,
        )

        # self.plot_and_log_ratemaps(
        #     rate_maps=rate_maps,
        #     score_60_by_neuron=score_60_by_neuron,
        #     score_90_by_neuron=score_90_by_neuron,
        #     epoch_idx=epoch_idx,
        #     log_to_wandb=log_to_wandb,
        #     run_dir=run_dir)

        # results = dict(
        #     rate_maps=rate_maps,
        #     score_60_by_neuron=score_60_by_neuron,
        #     score_90_by_neuron=score_90_by_neuron,
        #     period_per_cell=period_per_cell,
        #     period_err_per_cell=period_err_per_cell,
        #     orientations_per_cell=orientations_per_cell,
        # )
        #
        # return results

    def compute_intrinsic_dimensionalities(self,
                                           inputs):
        # activations has shape: (batch_size * sequence_length, Ng)
        activations = tf.reshape(
            tf.stop_gradient(self.model.g(inputs)),
            shape=[self.options.batch_size * self.options.sequence_length, self.options.Ng]
        )

        # TODO: add ID measures to rate maps (spatial vs neuron)

        # In this function, ID stands for Intrinsic Dimensionality.
        try:
            two_NN_ID = skdim.id.TwoNN().fit_transform(X=activations)
        except ValueError:
            two_NN_ID = np.nan

        try:
            method_of_moments_ID = skdim.id.MOM().fit_transform(X=activations)
        except ValueError:
            method_of_moments_ID = np.nan

        # Use skdim implementation for trustworthiness.
        try:
            participation_ratio_ID = skdim.id.lPCA(ver='participation_ratio').fit_transform(
                X=activations)
        except ValueError:
            participation_ratio_ID = np.nan

        intrinsic_dimensionalities = dict(
            participation_ratio=participation_ratio_ID,
            two_NN=two_NN_ID,
            method_of_moments_ID=method_of_moments_ID,
        )

        return intrinsic_dimensionalities

    def compute_and_log_grid_cell_periodicity_and_orientation(self,
                                                              rate_maps_and_scores_dict: Dict[str, np.ndarray],
                                                              epoch_idx: int,
                                                              run_dir: str,
                                                              threshold: float = 0.3,
                                                              log_to_wandb: bool = True,
                                                              refresh: bool = False):

        period_results_joblib_path = os.path.join(run_dir, 'period_results_path.joblib')
        if refresh or not os.path.isfile(period_results_joblib_path):

            print('Computing grid cell periodicity and orientation.')

            values_to_dump = {
                'threshold': threshold
            }

            for scorer in self.scorers:
                nbins = scorer._nbins
                rate_maps = rate_maps_and_scores_dict[f'rate_maps_nbins={nbins}']
                score_60_by_neuron = rate_maps_and_scores_dict[f'score_60_by_neuron_nbins={nbins}']

                period_per_cell, period_err_per_cell, orientations_per_cell = [], [], []
                likely_grid_cell_indices = score_60_by_neuron > threshold
                for rate_map_idx, rate_map in enumerate(rate_maps):
                    if likely_grid_cell_indices[rate_map_idx]:
                        rate_map_copy = np.copy(rate_map)
                        # NOTE: This is new as of 2022/04/19.
                        rate_map_copy[np.isnan(rate_map_copy)] = 0.
                        period, period_err, orientations = scorer.calculate_grid_cell_periodicity_and_orientation(
                            rate_map=rate_map_copy)
                        period_per_cell.append(period)
                        period_err_per_cell.append(period_err)
                        orientations_per_cell.append(orientations.tolist())
                    else:
                        period_per_cell.append(np.nan)
                        period_err_per_cell.append(np.nan)
                        orientations_per_cell.append(np.nan)

                # if log_to_wandb:
                #     wandb.log({
                #         f'period_per_cell_threshold={threshold}': period_per_cell,
                #         f'period_err_per_cell_threshold={threshold}': period_err_per_cell,
                #         f'orientations_per_cell_threshold={threshold}': orientations_per_cell,
                #     }, step=epoch_idx + 1)

                values_to_dump.update({
                    f'period_per_cell_nbins={nbins}': period_per_cell,
                    f'period_err_per_cell_nbins={nbins}': period_err_per_cell,
                    f'orientations_per_cell_nbins={nbins}': orientations_per_cell})

            joblib.dump(
                values_to_dump,
                filename=period_results_joblib_path
            )

            print('Wrote computed grid cell periodicity and orientation to disk.')
        else:
            previously_generated_results = joblib.load(period_results_joblib_path)
            previous_threshold = previously_generated_results['threshold']
            if previous_threshold != threshold:
                raise ValueError(
                    f'Previous threshold ({previous_threshold}) does not equal desired threshold ({threshold}).')
            period_per_cell = previously_generated_results['period_per_cell']
            period_err_per_cell = previously_generated_results['period_err_per_cell']
            orientations_per_cell = previously_generated_results['orientations_per_cell']
            print('Loaded previously computed grid cell periodicity and orientation from disk.')

        # return period_per_cell, period_err_per_cell, orientations_per_cell

    def compute_and_log_rate_maps(self,
                                  xs,
                                  ys,
                                  activations,
                                  epoch_idx: int,
                                  n_recurr_units_to_analyze: int,
                                  run_dir: str,
                                  log_to_wandb: bool = True,
                                  refresh: bool = False):

        rate_maps_and_scores_joblib_path = os.path.join(run_dir, 'rate_maps_and_scores.joblib')
        if refresh or not os.path.isfile(rate_maps_and_scores_joblib_path):
            print('Creating rate maps and grid scores.')

            rate_maps_and_scores_dict = {}

            for scorer in self.scorers:

                # Create scores and ratemaps, then save to dis.
                score_60_by_neuron = np.zeros(n_recurr_units_to_analyze)
                score_90_by_neuron = np.zeros(n_recurr_units_to_analyze)

                best_score_60 = -np.inf
                best_rate_map_60 = None
                best_score_90 = -np.inf
                best_rate_map_90 = None

                neuron_indices = np.random.choice(self.options.Ng, replace=False,
                                                  size=n_recurr_units_to_analyze)

                rate_maps = np.zeros(shape=(n_recurr_units_to_analyze, scorer._nbins, scorer._nbins))

                for storage_idx, neuron_idx in enumerate(neuron_indices):

                    print(f'Generating rate map for neuron: {neuron_idx}')

                    # Would recommend switching to visualize.py's `compute_and_log_rate_maps`
                    # Then util.py's `get_model_activations`
                    # then utils.py's get_model_gridscores
                    #    model_resp is the rate map for a single layer
                    rate_map = scorer.calculate_ratemap(
                        xs=xs,
                        ys=ys,
                        activations=activations[:, neuron_idx],
                    )

                    print(f'Generated rate map for neuron: {neuron_idx}')

                    scores = scorer.get_scores(rate_map=rate_map)
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

                    print(f'Neuron: {storage_idx}\tScore60: {np.round(score_60, 3)}\tScore90: {np.round(score_90, 3)}')

                nbins = scorer._nbins

                rate_maps_and_scores_dict.update({
                    f'rate_maps_nbins={nbins}': rate_maps,
                    f'score_60_by_neuron_nbins={nbins}': score_60_by_neuron,
                    f'score_90_by_neuron_nbins={nbins}': score_90_by_neuron})

            joblib.dump(
                rate_maps_and_scores_dict,
                filename=rate_maps_and_scores_joblib_path
            )

            # # W&B recommends saving
            # if log_to_wandb:
            #     # These NaNs need to be zeroed out if logging as heatmaps to W&B,
            #     # otherwise W&B throws an error.
            #     best_rate_map_60[np.isnan(best_rate_map_60)] = 0.
            #     best_rate_map_90[np.isnan(best_rate_map_90)] = 0.
            #
            #     wandb.log({
            #         f'max_grid_score_d=60_n={n_recurr_units_to_analyze}': np.nanmax(score_60_by_neuron),
            #         f'grid_score_histogram_d=60_n={n_recurr_units_to_analyze}': wandb.Histogram(
            #             score_60_by_neuron[~np.isnan(score_60_by_neuron)]),
            #         f'best_rate_map_d=60_n={n_recurr_units_to_analyze}': best_rate_map_60,
            #         f'max_grid_score_d=90_n={n_recurr_units_to_analyze}': np.nanmax(score_90_by_neuron),
            #         f'grid_score_histogram_d=90_n={n_recurr_units_to_analyze}': wandb.Histogram(
            #             score_90_by_neuron[~np.isnan(score_90_by_neuron)]),
            #         f'best_rate_map_d=90_n={n_recurr_units_to_analyze}': best_rate_map_90,
            #     }, step=epoch_idx + 1)

        else:
            rate_maps_and_scores_dict = joblib.load(rate_maps_and_scores_joblib_path)

        assert len(rate_maps_and_scores_dict['rate_maps_nbins=20']) == len(
            rate_maps_and_scores_dict['score_60_by_neuron_nbins=20'])
        assert len(rate_maps_and_scores_dict['rate_maps_nbins=20']) == len(
            rate_maps_and_scores_dict['score_90_by_neuron_nbins=20'])

        return rate_maps_and_scores_dict

    def plot_and_log_ratemaps(self,
                              rate_maps,
                              score_60_by_neuron,
                              score_90_by_neuron,
                              epoch_idx: int,
                              log_to_wandb: bool = True,
                              run_dir: str = None):

        n_recurr_units_to_analyze = rate_maps.shape[0]
        n_rows = n_cols = int(np.ceil(np.sqrt(n_recurr_units_to_analyze)))

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
