# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import scipy

class PlaceCells(object):
    def __init__(self, options):
        self.place_cell_identity = options.place_cell_identity
        self.Np = options.Np
        self.banino_place_cell = options.banino_place_cell
        if self.place_cell_identity:
            assert(self.Np == 2)
        self.sigma = options.place_cell_rf
        self.surround_scale = options.surround_scale
        self.min_x = options.min_x
        self.max_x = options.max_x
        self.min_y = options.min_y
        self.max_y = options.max_y
        self.is_periodic = options.periodic
        self.vr1d = hasattr(options, 'vr1d') and (options.vr1d is True)
        self.DoG = options.DoG

        # Randomly tile place cell centers across environment
        tf.random.set_seed(0)
        usx = tf.random.uniform((self.Np,), self.min_x, self.max_x, dtype=tf.float64)
        if self.vr1d:
            assert(self.min_y == self.max_y)
            usy = self.min_y*tf.ones((self.Np,), dtype=tf.float64)
        else:
            usy = tf.random.uniform((self.Np,), self.min_y, self.max_y, dtype=tf.float64)
        self.us = tf.stack([usx, usy], axis=-1)
        if self.banino_place_cell:
            # Create a random MoG with fixed cov over the position (Nx2)
            rs = np.random.RandomState(None)
            self.variances = tf.ones_like(self.us) * (self.sigma**2)

    def get_activation(self, pos):
        '''
        Get place cell activations for a given position.

        Args:
            pos: 2d position of shape [batch_size, sequence_length, 2].

        Returns:
            outputs: Place cell activations with shape [batch_size, sequence_length, Np].
        '''
        if self.place_cell_identity:
            outputs = tf.cast(tf.identity(pos), dtype=tf.float32)
        else:
            d = tf.abs(pos[:, :, tf.newaxis, :] - self.us[tf.newaxis, tf.newaxis, ...])

            #if self.is_periodic:
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

            if self.banino_place_cell:
                assert(not self.DoG)
                logp = -0.5 * tf.reduce_sum((d**2)/ self.variances, axis=-1)
                outputs = logp - tf.reduce_logsumexp(logp, axis=2, keepdims=True)
                outputs = tf.nn.softmax(outputs, axis=-1)
            else:
                norm2 = tf.reduce_sum(d**2, axis=-1)

                # Normalize place cell outputs with prefactor alpha=1/2/np.pi/self.sigma**2,
                # or, simply normalize with softmax, which yields same normalization on
                # average and seems to speed up training.
                outputs = tf.nn.softmax(-norm2/(2*self.sigma**2))

                if self.DoG:
                    # Again, normalize with prefactor
                    # beta=1/2/np.pi/self.sigma**2/self.surround_scale, or use softmax.
                    outputs -= tf.nn.softmax(-norm2/(2*self.surround_scale*self.sigma**2))

                    # Shift and scale outputs so that they lie in [0,1].
                    outputs += tf.abs(tf.reduce_min(outputs, axis=-1, keepdims=True))
                    outputs /= tf.reduce_sum(outputs, axis=-1, keepdims=True)
        return outputs


    def get_nearest_cell_pos(self, activation, k=3):
        '''
        Decode position using centers of k maximally active place cells.

        Args:
            activation: Place cell activations of shape [batch_size, sequence_length, Np].
            k: Number of maximally active place cells with which to decode position.

        Returns:
            pred_pos: Predicted 2d position with shape [batch_size, sequence_length, 2].
        '''
        if self.place_cell_identity:
            pred_pos = tf.cast(tf.identity(activation), dtype=tf.float32)
        else:
            _, idxs = tf.math.top_k(activation, k=k)
            pred_pos = tf.reduce_mean(tf.gather(self.us, idxs), axis=-2)
        return pred_pos


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
