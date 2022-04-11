# -*- coding: utf-8 -*-
import numpy as np
import scipy
import tensorflow as tf


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
        self.n_recurrent_units_to_sample = None
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
        assert self.activation in {'relu', 'tanh', 'sigmoid'}
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
            'crossentropy'}
        self.place_field_loss = options.place_field_loss

        assert options.place_field_values in {
            'cartesian',
            'gaussian',
            'difference_of_gaussians'}
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
        self.n_place_fields_per_cell = float(options.n_place_fields_per_cell)
        self.sigma = options.place_cell_rf
        self.surround_scale = options.surround_scale
        self.min_x = options.min_x
        self.max_x = options.max_x
        self.min_y = options.min_y
        self.max_y = options.max_y

        self.place_field_loss = options.place_field_loss
        self.place_cell_rf = options.place_cell_rf

        self.is_periodic = options.is_periodic
        self.vr1d = hasattr(options, 'vr1d') and (options.vr1d is True)

        # Randomly tile place cell centers across environment
        max_n_place_fields_per_cell = int(np.ceil(self.n_place_fields_per_cell) + 1)
        usx = tf.random.uniform(
            shape=(self.Np, int(np.ceil(self.n_place_fields_per_cell)) + 1),
            minval=self.min_x,
            maxval=self.max_x,
            dtype=tf.float64)
        if self.vr1d:
            assert (self.min_y == self.max_y)
            usy = self.min_y * tf.ones((self.Np,), dtype=tf.float64)
        else:
            usy = tf.random.uniform(
                shape=(self.Np, max_n_place_fields_per_cell),
                minval=self.min_y,
                maxval=self.max_y,
                dtype=tf.float64)

        # Shape: (Num place cells, num fields per cell, 2 for XY)
        self.us = tf.stack([usx, usy], axis=-1)

        # If num fields per cell is not a whole integer, we want to randomly
        # "delete" fields. Since this is difficult with fixed-sized arrays, the
        # simplest workaround is to set a random subset to ridiculously far
        # away values.
        fields_to_delete = tf.random.uniform(
            shape=(self.Np, max_n_place_fields_per_cell),
            minval=0.,
            maxval=1.0,
            dtype=tf.float64) > (self.n_place_fields_per_cell / max_n_place_fields_per_cell)
        fields_to_delete = tf.cast(fields_to_delete, dtype=tf.float64)
        # Rather than deleting, just move the fields far far away. By setting the locations
        # to a ridiculous value, these place fields will never be active.
        # Rather than Just move the fields far, far away. 1e5 is a heuristic.
        replacement_locations_for_fields_to_delete = 1e5 * self.us
        # Irritatingly, TensorFlow doesn't permit assigning to the LHS (see
        # https://stackoverflow.com/a/62472890), so we have to use this complicated workaround.
        self.us = tf.add(
            tf.multiply(fields_to_delete[:, :, tf.newaxis], replacement_locations_for_fields_to_delete),
            tf.multiply(1 - fields_to_delete[:, :, tf.newaxis], self.us),
        )

    def get_activation(self, pos):
        '''
        Get place cell activations for a given position.
        Args:
            pos: 2d position of shape [batch_size, sequence_length, 2].
        Returns:
            outputs: Place cell activations with shape [batch_size, sequence_length, Np].
        '''
        if self.place_field_values == 'cartesian':
            outputs = tf.cast(tf.identity(pos), dtype=tf.float64)
            return outputs
        # Shape: (batch size, sequence length, num place cells, num fields per cell, 2)
        d = tf.abs(pos[:, :, tf.newaxis, tf.newaxis, :] - self.us[tf.newaxis, tf.newaxis, ...])

        # Take the place field per cell closest to the position.
        d = tf.reduce_min(d, axis=3)

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

        norm2 = tf.reduce_sum(d ** 2, axis=-1)

        transformed_norm2 = self.normalize_norm2(
            norm2=norm2,
            place_field_normalization=self.place_field_normalization,
            dividing_scalar=2 * self.place_cell_rf ** 2)

        if self.place_field_values == 'gaussian':
            outputs = transformed_norm2
        elif self.place_field_values == 'difference_of_gaussians':

            other_transformed_norm2 = self.normalize_norm2(
                norm2=norm2,
                place_field_normalization=self.place_field_normalization,
                dividing_scalar=2 * self.surround_scale * self.place_cell_rf ** 2)

            diff_of_transformed = transformed_norm2 - other_transformed_norm2

            # Shift and scale outputs so that they lie in [0,1].
            diff_of_transformed += tf.abs(tf.reduce_min(diff_of_transformed, axis=-1, keepdims=True))
            diff_of_transformed /= tf.reduce_sum(diff_of_transformed, axis=-1, keepdims=True)
            outputs = diff_of_transformed
        else:
            raise ValueError(f"Impermissible place field function: {self.place_field_loss}")

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
        if self.place_field_values == 'cartesian':
            pred_pos = tf.cast(tf.identity(activation), dtype=tf.float64)
        else:
            # Shape: (batch size, sequence length, Np)
            # Original.
            # _, idxs = tf.math.top_k(activation, k=k)

            # For some reason, activation is float32. Recast it to 64.
            activation = tf.cast(activation, dtype=tf.float64)

            # Recall, self.us has shape (Np, num fields per place cell, 2)
            # and activation has shape (batch size, sequence length, Np)
            pred_pos = tf.reduce_mean(tf.multiply(
                activation[:, :, :, tf.newaxis, tf.newaxis],  # add 2 dimensions for fields/cell and for cartesian coordinates
                self.us[tf.newaxis, tf.newaxis, :, :, :],  # add 2 dimensions for batch size and sequence length
            ), axis=(2, 3))

        return pred_pos

    @staticmethod
    def normalize_norm2(norm2: tf.Tensor,
                        dividing_scalar: float,
                        place_field_normalization: str
                        ) -> tf.Tensor:

        if place_field_normalization == 'local':
            output = tf.exp(-norm2 / dividing_scalar)
        elif place_field_normalization == 'global':
            output = tf.nn.softmax(-norm2 / dividing_scalar)
        else:
            raise ValueError(f"Impermissible normalization: {place_field_normalization}")
        return output

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
