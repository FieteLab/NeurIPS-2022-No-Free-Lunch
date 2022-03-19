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
        self.place_cell_rf = None
        self.place_field_function = None
        self.place_field_normalization = None
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
        assert self.place_field_function in {'position',
                                             'gaussian_local',
                                             'gaussian_global',
                                             'difference_of_gaussians_local',
                                             'difference_of_gaussians_global'}
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
        assert(self.Nhdc is not None)
        self.concentration = options.hdc_concentration
        # Create a random Von Mises with fixed cov over the position
        rs = np.random.RandomState(None)
        self.means = rs.uniform(-np.pi, np.pi, (self.Nhdc))
        self.kappa = np.ones_like(self.means) * self.concentration

    def get_activation(self, x):
        assert(len(x.shape) == 2)
        logp = self.kappa * tf.cos(x[:, :, np.newaxis] - self.means[np.newaxis, np.newaxis, :])
        outputs = logp - tf.reduce_logsumexp(logp, axis=2, keepdims=True)
        outputs = tf.nn.softmax(outputs, axis=-1)
        return outputs


class PlaceCells(object):
    def __init__(self,
                 options: Options):

        assert options.place_field_function in {
            'position',
            'gaussian',
            'difference_of_gaussians'}
        self.place_field_function = options.place_field_function
        assert options.place_field_normalization in {
            'none',
            'normal',
            'softmax',
        }
        self.place_field_normalization = options.place_field_normalization

        if self.place_field_function == 'position':
            assert options.Np == 2

        self.Np = options.Np
        self.sigma = options.place_cell_rf
        self.surround_scale = options.surround_scale
        self.min_x = options.min_x
        self.max_x = options.max_x
        self.min_y = options.min_y
        self.max_y = options.max_y

        self.place_field_function = options.place_field_function
        self.place_cell_rf = options.place_cell_rf

        self.is_periodic = options.is_periodic
        self.vr1d = hasattr(options, 'vr1d') and (options.vr1d is True)

        # Randomly tile place cell centers across environment
        tf.random.set_seed(0)
        usx = tf.random.uniform((self.Np,), self.min_x, self.max_x, dtype=tf.float64)
        if self.vr1d:
            assert (self.min_y == self.max_y)
            usy = self.min_y * tf.ones((self.Np,), dtype=tf.float64)
        else:
            usy = tf.random.uniform((self.Np,), self.min_y, self.max_y, dtype=tf.float64)
        self.us = tf.stack([usx, usy], axis=-1)

    def get_activation(self, pos):
        '''
        Get place cell activations for a given position.
        Args:
            pos: 2d position of shape [batch_size, sequence_length, 2].
        Returns:
            outputs: Place cell activations with shape [batch_size, sequence_length, Np].
        '''
        if self.place_field_function == 'position':
            outputs = tf.cast(tf.identity(pos), dtype=tf.float32)
            return outputs

        d = tf.abs(pos[:, :, tf.newaxis, :] - self.us[tf.newaxis, tf.newaxis, ...])

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

        if self.place_field_function == 'gaussian':
            outputs = transformed_norm2
        elif self.place_field_function == 'difference_of_gaussians':

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
            raise ValueError(f"Impermissible place field function: {self.place_field_function}")

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
        if self.place_field_function == 'position':
            pred_pos = tf.cast(tf.identity(activation), dtype=tf.float32)
        else:
            _, idxs = tf.math.top_k(activation, k=k)
            pred_pos = tf.reduce_mean(tf.gather(self.us, idxs), axis=-2)
        return pred_pos

    @staticmethod
    def normalize_norm2(norm2: tf.Tensor,
                        dividing_scalar: float,
                        place_field_normalization: str
                        ) -> tf.Tensor:

        if place_field_normalization == 'none':
            output = tf.exp(-norm2 / dividing_scalar)
        elif place_field_normalization == 'normal':
            output = tf.exp(-norm2 / dividing_scalar)
            output /= np.pi * dividing_scalar
        elif place_field_normalization == 'softmax':
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

    #     pos = pos.astype(np.float32)

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
