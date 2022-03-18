# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

class HeadDirectionCells(object):
    def __init__(self, options):
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
