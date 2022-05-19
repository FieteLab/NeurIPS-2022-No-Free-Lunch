# -*- coding: utf-8 -*-
from itertools import product
import numpy as np
import scipy
import tensorflow as tf
from typing import Dict, List, Tuple, Union


class Options(object):

    def __init__(self):
        self.activation = None
        self.batch_size = None
        self.bin_side_in_m = None
        self.box_width_in_m = None
        self.box_height_in_m = None
        self.learning_rate = None
        self.initializer = None
        self.is_periodic = None
        self.max_x = None
        self.min_x = None
        self.max_y = None
        self.min_y = None
        self.Np = None
        self.n_epochs = None
        self.n_grad_steps_per_epoch = None
        self.n_place_fields_per_cell = None
        self.Ng = None
        self.optimizer = None
        self.place_field_loss = None
        self.place_field_values = None
        self.place_field_normalization = None
        self.place_cell_rf = None
        self.readout_dropout = None
        self.recurrent_dropout = None
        self.rnn_type = None
        self.seed = None
        self.sequence_length = None
        self.surround_scale = None
        self.weight_decay = None

    def check_valid(self):
        assert self.activation in {'relu', 'tanh', 'sigmoid', 'linear'}
        self.assert_var_is_positive_int(self.batch_size)
        assert self.place_field_loss in {'cartesian',
                                         'gaussian',
                                         'difference_of_gaussians'}
        self.assert_var_is_positive_float(self.surround_scale)

        assert isinstance(self.rnn_type, str)
        assert 0 <= self.readout_dropout <= 1.
        assert 0 <= self.recurrent_dropout <= 1.
        assert self.rnn_type in {'RNN', 'LSTM', 'GRU', 'UGRNN'}

    @staticmethod
    def assert_var_is_positive_int(v):
        assert isinstance(v, int) and v > 0

    @staticmethod
    def assert_var_is_positive_float(v):
        assert isinstance(v, float) and v > 0.


class HeadDirectionCells(object):
    def __init__(self,
                 options: Options):
        self.Nhdc = options.Nhdc
        assert (self.Nhdc is not None)
        self.concentration = options.hdc_concentration
        # Create a random Von Mises with fixed cov over the position
        rs = np.random.RandomState(None)
        self.means = rs.uniform(-np.pi, np.pi, (self.Nhdc))
        self.kappa = np.ones_like(self.means) * self.concentration

    def get_activation(self, x):
        assert (len(x.shape) == 2)
        logp = self.kappa * tf.cos(x[:, :, np.newaxis] - self.means[np.newaxis, np.newaxis, :])
        outputs = logp - tf.reduce_logsumexp(logp, axis=2, keepdims=True)
        outputs = tf.nn.softmax(outputs, axis=-1)
        return outputs


class PlaceCells(object):

    def __init__(self,
                 options: Options):

        assert options.place_field_loss in {
            'mse',
            'polarmse',
            'crossentropy',
            'binarycrossentropy'}
        self.place_field_loss = options.place_field_loss

        assert options.place_field_values in {
            'cartesian',
            'polar',
            'gaussian',
            'difference_of_gaussians',
            'true_difference_of_gaussians',
            'softmax_of_differences',
        }
        self.place_field_values = options.place_field_values

        if self.place_field_values == 'cartesian':
            assert options.Np == 2

        assert options.place_field_normalization in {
            'none',
            'local',
            'global',
        }
        self.place_field_normalization = options.place_field_normalization

        self.Np = options.Np
        self.place_field_loss = options.place_field_loss
        self.min_x = options.min_x
        self.max_x = options.max_x
        self.min_y = options.min_y
        self.max_y = options.max_y

        self.is_periodic = options.is_periodic
        self.vr1d = hasattr(options, 'vr1d') and (options.vr1d is True)

        # Choose locations for place cells.
        self.n_place_fields_per_cell = options.n_place_fields_per_cell
        if isinstance(options.n_place_fields_per_cell, int):

            self.max_n_place_fields_per_cell = self.n_place_fields_per_cell
            usx = tf.random.uniform(
                shape=(self.Np, self.max_n_place_fields_per_cell, 1),
                minval=self.min_x,
                maxval=self.max_x,
                dtype=tf.float32)
            usy = tf.random.uniform(
                shape=(self.Np, self.max_n_place_fields_per_cell, 1),
                minval=self.min_y,
                maxval=self.max_y,
                dtype=tf.float32)
            # Shape (Np, max fields per cell, 2)
            self.us = tf.concat([usx, usy], axis=2)
            self.fields_to_delete = tf.zeros(
                shape=(self.Np, self.max_n_place_fields_per_cell),
                dtype=tf.float32)

        elif isinstance(options.n_place_fields_per_cell, str):

            if options.n_place_fields_per_cell.startswith('Poisson'):
                rate = self.extract_floats_from_str(options.n_place_fields_per_cell)
                n_fields_per_cell = 1 + tf.random.poisson(
                    shape=(self.Np,),
                    lam=rate,
                    dtype=tf.float32)  # Add 1 to ensures that each cell has at least 1 field.
                self.max_n_place_fields_per_cell = int(tf.reduce_max(n_fields_per_cell))

                usx = tf.random.uniform(
                    shape=(self.Np, self.max_n_place_fields_per_cell, 1),
                    minval=self.min_x,
                    maxval=self.max_x,
                    dtype=tf.float32)
                usy = tf.random.uniform(
                    shape=(self.Np, self.max_n_place_fields_per_cell, 1),
                    minval=self.min_y,
                    maxval=self.max_y,
                    dtype=tf.float32)
                # Shape (Np, max fields per cell, 2)
                self.us = tf.concat([usx, usy], axis=2)

                # Shape: (num place cells, max num fields per cell)
                # Create array of indices
                fields_to_delete = tf.repeat(
                    tf.range(self.max_n_place_fields_per_cell, dtype=tf.float32)[tf.newaxis, :],
                    repeats=self.Np,
                    axis=0)
                fields_to_delete = fields_to_delete >= n_fields_per_cell

            else:
                raise NotImplementedError

            # Shape: (num place cells, max num fields per cell, 1)
            self.fields_to_delete = tf.cast(fields_to_delete, dtype=tf.float32)[:, :, tf.newaxis]
            # Rather than deleting, move the fields far away. By setting the locations
            # to a ridiculous value, these place fields will never be active.
            # 1e7 is a heuristic.
            replacement_locations_for_fields_to_delete = 1e7 * self.us
            # Irritatingly, TensorFlow doesn't permit assigning to the LHS (see
            # https://stackoverflow.com/a/62472890), so we have to use this complicated workaround.
            self.us = tf.add(
                tf.multiply(self.fields_to_delete, replacement_locations_for_fields_to_delete),
                tf.multiply(1. - self.fields_to_delete, self.us),
            )
        else:
            raise NotImplementedError

        self.fields_to_keep = 1. - self.fields_to_delete

        # if self.vr1d:
        #     # assert (self.min_y == self.max_y)
        #     # usy = self.min_y * tf.ones((self.Np,), dtype=tf.float32)
        #     raise NotImplementedError
        # else:
        #     usy = tf.random.uniform(
        #         shape=(self.Np, max_n_place_fields_per_cell),
        #         minval=self.min_y,
        #         maxval=self.max_y,
        #         dtype=tf.float32)
        #
        # # Shape: (Num place cells, num fields per cell, 2 for XY)
        # self.us = tf.stack([usx, usy], axis=-1)

        # Create place cell receptive field tensor.
        if isinstance(options.place_cell_rf, (float, int)):
            # Add the 1, 1, to the shape for future broadcasting
            self.place_cell_rf = float(options.place_cell_rf) * tf.ones(
                shape=(1, 1, self.Np, self.max_n_place_fields_per_cell),
                dtype=tf.float32)
        elif isinstance(options.place_cell_rf, str):
            if options.place_cell_rf.startswith('Uniform'):
                low, high = self.extract_floats_from_str(s=options.place_cell_rf)
                # Add the 1, 1, to the shape for future broadcasting
                self.place_cell_rf = tf.random.uniform(
                    shape=(1, 1, self.Np, self.max_n_place_fields_per_cell),
                    minval=low,
                    maxval=high,
                    dtype=tf.float32)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # Create second place cell receptive field tensor.
        if isinstance(options.surround_scale, (float, int)):
            # Add the 1, 1, to the shape for future broadcasting
            self.surround_scale = float(options.surround_scale) * tf.ones(
                shape=(1, 1, self.Np, self.max_n_place_fields_per_cell),
                dtype=tf.float32)
        elif isinstance(options.surround_scale, str):
            if options.surround_scale.startswith('Uniform'):
                # Add the 1, 1, to the shape for future broadcasting
                low, high = self.extract_floats_from_str(s=options.surround_scale)
                self.surround_scale = tf.random.uniform(
                    shape=(1, 1, self.Np, self.max_n_place_fields_per_cell),
                    minval=low,
                    maxval=high,
                    dtype=tf.float32)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        # Ensure that surround receptive field is greater than receptive field.
        if self.place_field_values == 'difference_of_gaussians':
            assert tf.reduce_min(self.surround_scale) > 1.

    def get_activation(self, pos):
        '''
        Get place cell activations for a given position.
        Args:
            pos: 2d position of shape [batch_size, sequence_length, 2].
        Returns:
            outputs: Place cell activations with shape [batch_size, sequence_length, Np].
        '''
        if self.place_field_values == 'cartesian':
            outputs = tf.cast(tf.identity(pos), dtype=tf.float32)
            return outputs

        if self.place_field_values == 'polar':
            # Shape: (batch size, seq len, 1)
            r = tf.math.sqrt(tf.reduce_sum(tf.math.square(pos), axis=2, keepdims=True))
            # Shape: (batch size, seq len, 1)
            theta = tf.math.atan(pos[..., 1, tf.newaxis] / pos[..., 0, tf.newaxis])
            # Shape: (batch size, seq len, 2)
            outputs = tf.concat([r, theta], axis=2)
            return outputs

        # Shape: (batch size, sequence length, num place cells, max num fields per cell, 2)
        d = tf.abs(pos[:, :, tf.newaxis, tf.newaxis, :] - self.us[tf.newaxis, tf.newaxis, ...])

        # # Take the place field per cell closest to the position.
        # d = tf.reduce_min(d, axis=3)

        # if self.is_periodic:
        #    dx = tf.gather(d, 0, axis=-1)
        #    dy = tf.gather(d, 1, axis=-1)
        #    before_min = dx
        #    # this minimum won't change anything unless the position variable goes past the boundary max_x
        #    dx = tf.minimum(dx, (self.max_x - self.min_x) - dx)
        #    after_min = dx
        #    print("before: ", (self.max_x - self.min_x), tf.reduce_max(before_min))
        #    print("after: ", (self.max_x - self.min_x), tf.reduce_max(after_min))
        #    dy = tf.minimum(dy, (self.max_y - self.min_y) - dy)
        #    d = tf.stack([dx,dy], axis=-1)

        # Compute distance over 2D cartesian position
        # Shape: (batch size, sequence length, num place cells, num fields per cell)
        dist_squared = tf.reduce_sum(d ** 2, axis=4)
        divided_dist_squared = tf.divide(
            dist_squared,
            2.*tf.square(self.place_cell_rf),  # shape: (1, 1, num place cells, max num fields per cell)
        )
        # Shape: (batch size, trajectory length, num place cells)
        min_indices = tf.math.argmin(
            divided_dist_squared,
            axis=3)
        # Shape: (batch size, trajectory length, num place cells)
        min_divided_dist_squared = tf.gather(divided_dist_squared, min_indices, batch_dims=3)

        # TODO: implement local normalization
        if self.place_field_normalization == 'none':
            normalized_dist_squared = tf.exp(-min_divided_dist_squared)
        elif self.place_field_normalization == 'global':
            normalized_dist_squared = tf.nn.softmax(-min_divided_dist_squared, axis=2)
        else:
            raise ValueError(f"Impermissible normalization: {self.place_field_normalization}")

        if self.place_field_values == 'gaussian':
            # Shape: (batch size, sequence length, num place cells,)
            outputs = normalized_dist_squared

        elif self.place_field_values == 'true_difference_of_gaussians':
            # Shape: (batch size, sequence length, num place cells, num fields per cell)
            other_divided_dist_squared = tf.divide(
                dist_squared,
                2. * tf.square(tf.multiply(self.place_cell_rf, self.surround_scale))
            )
            # Shape: (batch size, sequence length, num place cells)
            min_other_divided_dist_squared = tf.gather(
                other_divided_dist_squared,
                min_indices,
                batch_dims=3)

            outputs = tf.math.exp(-min_divided_dist_squared) - tf.math.exp(-min_other_divided_dist_squared)

            # Original
            # Shift and scale outputs so that they lie in [0,1].
            outputs += tf.abs(tf.reduce_min(outputs, axis=-1, keepdims=True))
            outputs /= tf.reduce_sum(outputs, axis=-1, keepdims=True)

        elif self.place_field_values == 'softmax_of_differences':

            # Shape: (batch size, sequence length, num place cells, num fields per cell)
            other_divided_dist_squared = tf.divide(
                dist_squared,
                2. * tf.square(tf.multiply(self.place_cell_rf, self.surround_scale))
            )
            # Shape: (batch size, sequence length, num place cells)
            min_other_divided_dist_squared = tf.gather(
                other_divided_dist_squared,
                min_indices,
                batch_dims=3)

            outputs = tf.nn.softmax(-min_divided_dist_squared + min_other_divided_dist_squared, axis=2)

        elif self.place_field_values == 'difference_of_gaussians':

            # Shape: (batch size, sequence length, num place cells, num fields per cell)
            other_divided_dist_squared = tf.divide(
                dist_squared,
                2. * tf.square(tf.multiply(self.place_cell_rf, self.surround_scale))
            )
            # Shape: (batch size, sequence length, num place cells)
            min_other_divided_dist_squared = tf.gather(
                other_divided_dist_squared,
                min_indices,
                batch_dims=3)

            # TODO: implement local normalization
            if self.place_field_normalization == 'none':
                other_normalized_dist_squared = tf.exp(-min_other_divided_dist_squared)
            elif self.place_field_normalization == 'global':
                other_normalized_dist_squared = tf.nn.softmax(-min_other_divided_dist_squared, axis=2)
            else:
                raise ValueError(f"Impermissible normalization: {self.place_field_normalization}")

            # Shape: (batch size, sequence length, num place cells)
            outputs = normalized_dist_squared - other_normalized_dist_squared

            # Shift and scale outputs so that they lie in [0,1].
            outputs += tf.abs(tf.reduce_min(outputs, axis=-1, keepdims=True))
            outputs /= tf.reduce_sum(outputs, axis=-1, keepdims=True)

        else:
            raise ValueError(f"Impermissible place field function: {self.place_field_loss}")

        # Shape (batch size, seq length, num place cells,)
        return outputs

    def get_nearest_cell_pos(self,
                             activation,
                             k: int = 3):
        '''
        Decode position using centers of k maximally active place cells.
        Args:
            activation: Place cell activations of shape [batch_size, sequence_length, Np].
            k: Number of maximally active place cells with which to decode position.
        Returns:
            pred_pos: Predicted 2d position with shape [batch_size, sequence_length, 2].
        '''

        # TODO: implement this for a general number k
        assert k == 3

        if self.place_field_values == 'cartesian':
            pred_pos = tf.cast(tf.identity(activation), dtype=tf.float32)
        elif self.place_field_values == 'polar':
            # 0 is radius, 1 is theta
            # Shape: (batch size, seq len)
            pred_x = activation[:, :, 0] * tf.math.cos(activation[:, :, 1])
            # Shape: (batch size, seq len)
            pred_y = activation[:, :, 0] * tf.math.sin(activation[:, :, 1])
            # Shape: (batch size, seq len, 2)
            pred_pos = tf.stack([pred_x, pred_y], axis=2)
        else:
            # # Original:
            # _, idxs = tf.math.top_k(activation, k=k)
            # # Shape: (batch size, sequence length, Np)
            # # pred_pos = tf.reduce_mean(tf.gather(self.us, idxs), axis=-2)
            # # Warning: only works with 1 field per place field
            # pred_pos = tf.reduce_mean(tf.gather(tf.squeeze(self.us), idxs), axis=-2)

            # top_k applies to the last dimension.
            # Shape: (batch size, sequence length, k)
            _, top_k_indices = tf.math.top_k(activation, k=k)
            # Shape: (batch size, seq length, k, max fields per cell, 2 i.e. XY).
            # Ensure we can only gather the fields that we want to keep.
            voting_locations = tf.gather(self.us, top_k_indices)

            # Explanation: voting_locations has shape (batch size, seq length, K, max fields per cell, 2 i.e. XY).
            # Our approach is to consider all possible configurations of (max fields per cell)^K
            # We will construct a tensor of shape (batch size, seq length, 2, (max fields per cell)^K)
            # and then take the mean and variance over the last dimension. We will then return
            # the mean XY position of the configuration with the smallest variance.
            batch_size = voting_locations.shape[0]
            seq_len = voting_locations.shape[1]
            max_fields_per_cell = voting_locations.shape[3]
            # Shape: ((max fields per cell^K, max fields per cell,)
            indices_per_config = tf.convert_to_tensor(
                list(product(*[list(range(max_fields_per_cell)) for _ in range(k)])))
            # Shape: ((max fields per cell^K), max fields per cell, max fields per cell)
            # indices_one_hot = tf.one_hot(itf.floatndices_per_config, depth=self.max_n_place_fields_per_cell, axis=2, dtype=tf.float32)

            possible_configurations_locations = []
            for indices in indices_per_config:
                config = []
                for i, index in enumerate(indices):
                    config.append(voting_locations[:, :, i, index, :])
                # Shape: (batch size, seq length, 2, k)
                config = tf.stack(config, axis=-1)
                possible_configurations_locations.append(config)

            # Shape: (batch size, seq length, 2, k, total num configs = max fields per cell ^ k)
            possible_configurations_locations = tf.stack(possible_configurations_locations, axis=-1)

            # Shape: (batch size, seq length, 2, total num configs)
            possible_configurations_means, possible_configurations_variances = tf.nn.moments(
                possible_configurations_locations,
                axes=[3])
            indices_of_configuration_with_smallest_variance = tf.argmin(
                tf.reduce_sum(possible_configurations_variances, axis=2),
                axis=2)
            # Shape: (batch size, seq length, 2 i.e. XY, 1)
            pred_pos = tf.gather(
                possible_configurations_means,
                tf.repeat(indices_of_configuration_with_smallest_variance[:, :, tf.newaxis, tf.newaxis], 2, axis=2),
                batch_dims=-1)
            # Remove trailing dimension
            pred_pos = tf.squeeze(pred_pos, axis=3)

        return pred_pos

    @staticmethod
    def extract_floats_from_str(s: str) -> Tuple:
        # This assumes that the floats are separated by whitespace
        # e.g. Uniform( 0.5 , 3.5 )
        floats = []
        for sub_s in s.split():
            try:
                floats.append(float(sub_s))
            except ValueError:
                pass
        return tuple(floats)

    # def grid_pc(self, pc_outputs, res=32):
    #     ''' Interpolate place cell outputs onto a grid'''
    #     coordsx = np.linspace(self.min_x, self.max_x, res)
    #     coordsy = np.linspace(self.min_y, self.max_y, res)
    #     grid_x, grid_y = np.meshgrid(coordsx, coordsy)
    #     grid = np.stack([grid_x.ravel(), grid_y.ravel()]).T

    #     # Convert to numpy
    #     us_np = self.us.numpy()
    #     pc_outputs = pc_outputs.numpy().reshape(-1, self.Np)

    #     T = pc_outputs.shape[0] #T vs transpose? What is T? (dim's?)
    #     pc = np.zeros([T, res, res])
    #     for i in range(len(pc_outputs)):
    #         gridval = scipy.interpolate.griddata(us_np, pc_outputs[i], grid)
    #         pc[i] = gridval.reshape([res, res])

    #     return pc

    # def compute_covariance(self, res=30):
    #     '''Compute spatial covariance matrix of place cell outputs'''
    #     pos = np.array(np.meshgrid(np.linspace(self.min_x, self.max_x, res),
    #                      np.linspace(self.min_y, self.max_y, res))).T

    #     pos = pos.astype(np.float64)

    #     #Maybe specify dimensions here again?
    #     pc_outputs = self.get_activation(pos)
    #     pc_outputs = tf.reshape(pc_outputs, (-1, self.Np))

    #     C = pc_outputs@tf.transpose(pc_outputs)
    #     Csquare = tf.reshape(C, (res,res,res,res))

    #     Cmean = np.zeros([res,res])
    #     for i in range(res):
    #         for j in range(res):
    #             Cmean += np.roll(np.roll(Csquare[i,j], -i, axis=0), -j, axis=1)

    #     Cmean = np.roll(np.roll(Cmean, res//2, axis=0), res//2, axis=1)

    #     return Cmean
