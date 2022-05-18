# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, SimpleRNN, Layer
from tensorflow.keras.layers import RNN as RNN_wrapper
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
import wandb

from mec_hpc_investigations.models.helper_classes import Options, PlaceCells


def create_loss_fn(place_field_loss: str,
                   place_field_normalization: str):

    if place_field_loss == 'mse':
        loss_fn = pos_loss
    elif place_field_loss == 'polarmse':
        loss_fn = polar_mse_loss
    elif place_field_loss == 'crossentropy':
        if place_field_normalization == 'global':
            loss_fn = tf.nn.softmax_cross_entropy_with_logits
        else:
            raise ValueError(f'Impermissible normalization str: {place_field_normalization}')
    elif place_field_loss == 'binarycrossentropy':
        loss_fn = tf.nn.sigmoid_cross_entropy_with_logits
    else:
        raise ValueError(f'Impermissible place field loss str: {place_field_loss}')
    return loss_fn


def mask_func(inp, mask_mult=None, mask_add=None):
    if mask_add is not None:
        assert (mask_mult is not None)
    if mask_mult is not None:
        inp = inp * mask_mult
        if mask_add is not None:
            inp = inp + mask_add
    return inp


def pos_loss(x, y):
    return (x - y) ** 2


def polar_mse_loss(x, y):
    # x and y have shape (batch size, seq len, 2)
    # Compute: r_1^2 + r_2^2 - 2 r_1 r_2 cos(\theta_1 - \theta_2)
    # Dimension 0 is r, Dimension 1 is theta

    # All have shape: (batch size, seq len, 1)
    term1 = tf.math.square(x[..., 0, tf.newaxis])
    term2 = tf.math.square(y[..., 0, tf.newaxis])
    term3 = - 2. * tf.multiply(x[..., 0, tf.newaxis], y[..., 0, tf.newaxis]) * tf.math.cos(x[..., 1, tf.newaxis] - y[..., 1, tf.newaxis])
    return term1 + term2 + term3


class UGRNNCell(Layer):
    """Collins et al. 2017.
    Adapted from: https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/contrib/rnn/python/ops/rnn_cell.py#L1622-L1713"""

    def __init__(self,
                 units,
                 activation: str = "tanh",
                 initializer: str = 'glorot_uniform',
                 **kwargs):
        self.units = units
        self.state_size = units
        self.activation = activation
        self.initializer = initializer
        super(UGRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        """Initializations taken from here: https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/contrib/rnn/python/ops/core_rnn_cell.py#L126-L188"""
        self.weight = self.add_weight(shape=(input_shape[-1] + self.units, input_shape[-1] + self.units),
                                      initializer=self.initializer,
                                      name='weight',
                                      trainable=True)
        self.bias = self.add_weight(shape=(input_shape[-1] + self.units,),
                                    initializer='zeros',
                                    name='bias',
                                    trainable=True)
        self.built = True

    def call(self, inputs, states):
        prev_state = states[0]
        assert (prev_state.get_shape().as_list()[-1] == self.units)  # consistency
        input_dim = inputs.get_shape().as_list()[-1]
        assert (
                input_dim == self.units)  # otherwise elementwise multiply of g * prev_state in new_state update will fail
        cell_inputs = tf.concat([inputs, prev_state], axis=-1)
        rnn_matrix = tf.matmul(cell_inputs, self.weight) + self.bias
        [g_act, c_act] = tf.split(axis=-1, num_or_size_splits=[input_dim, self.units], value=rnn_matrix)
        c = getattr(tf.keras.activations, self.activation)(c_act)
        g = tf.nn.sigmoid(g_act + 1.0)
        new_state = g * prev_state + (1.0 - g) * c
        new_output = new_state
        return new_output, [new_state]


class RNN(Model):
    def __init__(self, options, place_cells):
        super(RNN, self).__init__()
        self.Ng = options.Ng
        self.Np = options.Np
        self.place_field_loss = options.place_field_loss
        self.place_field_normalization = options.place_field_normalization
        self.sequence_length = options.sequence_length
        self.weight_decay = options.weight_decay
        self.place_cells = place_cells

        self.encoder = Dense(self.Ng, name='encoder', use_bias=False)
        self.RNN = SimpleRNN(self.Ng,
                             return_sequences=True,
                             activation=tf.keras.layers.Activation(options.activation),
                             recurrent_initializer=options.initializer,
                             name='RNN',
                             use_bias=False)
        # Linear read-out weights
        self.decoder = Dense(self.Np, name='decoder', use_bias=False)

        # Loss function
        self.loss_fn = create_loss_fn(
            self.place_field_loss,
            self.place_field_normalization)

    def g(self, inputs):
        '''
        Compute grid cell activations.

        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2],
                    Along with the initial position to start integrating from.

        Returns:
            g: Batch of grid cell activations with shape [batch_size, sequence_length, Ng].
        '''
        v, p0 = inputs
        init_state = self.encoder(p0)
        g = self.RNN(v, initial_state=init_state)
        return g

    def dc(self, inputs, g_mask=None, g_mask_add=None):
        g = self.g(inputs)
        g = mask_func(inp=g, mask_mult=g_mask, mask_add=g_mask_add)
        return self.decoder(g)

    def call(self, inputs, g_mask=None, g_mask_add=None):
        '''
        Predict place cell code.

        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].

        Returns:
            place_preds: Predicted place cell activations with shape
                [batch_size, sequence_length, Np].
        '''
        place_preds = self.dc(inputs, g_mask=g_mask, g_mask_add=g_mask_add)

        return place_preds

    def compute_loss(self, inputs, pc_outputs, pos, g_mask=None, g_mask_add=None):
        '''
        Compute avg. loss and decoding error.

        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        '''
        preds = self.call(inputs, g_mask=g_mask, g_mask_add=g_mask_add)
        loss = tf.reduce_mean(self.loss_fn(pc_outputs, preds))

        # Weight regularization
        loss += self.weight_decay * tf.reduce_sum(self.RNN.weights[1] ** 2)

        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = tf.reduce_mean(tf.sqrt(tf.reduce_sum((tf.cast(pos, dtype=pred_pos.dtype) - pred_pos) ** 2, axis=-1)))

        return loss, err

    def log_weight_norms(self, epoch_idx: int):
        wandb_vals_to_log = {
            'input_matrix_norm': tf.reduce_sum(self.RNN.weights[0] ** 2).numpy(),
            'recurrent_matrix_norm': tf.reduce_sum(self.RNN.weights[1] ** 2).numpy(),
        }

        wandb.log(wandb_vals_to_log, step=epoch_idx+1)


class RewardRNN(RNN):
    '''
    Same as RNN, but gets rewards (or cues) as input.
    '''

    def __init__(self, **kwargs):
        super(RewardRNN, self).__init__(**kwargs)

    def g(self, inputs):
        v, p0, r = inputs
        init_state = self.encoder(p0)
        g = self.RNN(tf.concat([v, r], axis=-1), initial_state=init_state)
        return g


class LSTM(Model):
    def __init__(self,
                 options: Options,
                 place_cells: PlaceCells):
        super(LSTM, self).__init__()
        self.Ng = options.Ng
        self.Np = options.Np
        assert options.place_field_loss in {'mse',
                                            'binarycrossentropy',
                                            'crossentropy'}
        self.place_field_loss = options.place_field_loss
        self.place_field_normalization = options.place_field_normalization
        self.sequence_length = options.sequence_length
        self.weight_decay = options.weight_decay
        self.place_cells = place_cells

        self.encoder1 = Dense(self.Ng, name='encoder1')
        self.encoder2 = Dense(self.Ng, name='encoder2')
        self.M = Dense(self.Ng, name='M')
        self.RNN = tf.keras.layers.LSTM(self.Ng, return_sequences=True,
                                        activation=options.activation,
                                        recurrent_initializer=options.initializer)
        self.dense = Dense(self.Ng, name='dense', activation=options.activation)
        self.decoder = Dense(self.Np, name='decoder')

        # Loss function
        self.loss_fn = create_loss_fn(
            self.place_field_loss,
            self.place_field_normalization)

    def pre_g(self, inputs):
        '''Compute rnn cell activations'''
        v, p0 = inputs
        l0 = self.encoder1(p0)
        m0 = self.encoder2(p0)
        init_state = (l0, m0)
        Mv = self.M(v)
        rnn = self.RNN(Mv, initial_state=init_state)
        return rnn

    def g(self, inputs):
        '''Compute grid cell activations'''
        rnn = self.pre_g(inputs)
        g = self.dense(rnn)
        return g

    def dc(self, inputs, g_mask=None, g_mask_add=None):
        g = self.g(inputs)
        g = mask_func(inp=g, mask_mult=g_mask, mask_add=g_mask_add)
        return self.decoder(g)

    def call(self, inputs,
             g_mask=None, g_mask_add=None,
             dc_mask=None, dc_mask_add=None):
        '''Predict place cell code'''
        place_preds = self.dc(inputs, g_mask=g_mask, g_mask_add=g_mask_add)
        place_preds = mask_func(inp=place_preds, mask_mult=dc_mask, mask_add=dc_mask_add)
        return place_preds

    def compute_loss(self, inputs, pc_outputs, pos,
                     g_mask=None, g_mask_add=None,
                     dc_mask=None, dc_mask_add=None):
        '''Compute loss and decoding error'''
        preds = self.call(inputs,
                          g_mask=g_mask, g_mask_add=g_mask_add,
                          dc_mask=dc_mask, dc_mask_add=dc_mask_add)
        loss = tf.reduce_mean(self.loss_fn(pc_outputs, preds))

        # # Weight regularization
        loss += self.weight_decay * tf.reduce_sum(self.RNN.weights[1] ** 2)

        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = tf.reduce_mean(tf.sqrt(tf.reduce_sum((tf.cast(pos, dtype=pred_pos.dtype) - pred_pos) ** 2, axis=-1)))

        return loss, err

    def log_weight_norms(self, epoch_idx: int):
        wandb_vals_to_log = {
            'input_matrix_norm': tf.reduce_sum(self.RNN.weights[0] ** 2).numpy(),
            'recurrent_matrix_norm': tf.reduce_sum(self.RNN.weights[1] ** 2).numpy(),
            'bias_norm': tf.reduce_sum(self.RNN.weights[2] ** 2).numpy(),
        }

        wandb.log(wandb_vals_to_log, step=epoch_idx+1)


class RewardLSTM(LSTM):
    '''
    Same as LSTM, but gets rewards (or cues) as input.
    We don't encode the extra input passed to the underlying RNN,
    to mimic the GridCellAgent in DeepMind and the MERLIN paper.
    '''

    def __init__(self, **kwargs):
        super(RewardLSTM, self).__init__(**kwargs)

    def pre_g(self, inputs):
        v, p0, r = inputs
        l0 = self.encoder1(p0)
        m0 = self.encoder2(p0)
        init_state = (l0, m0)
        Mv = self.M(v)
        # so that concatenation will work
        r = tf.cast(r, dtype=Mv.dtype)
        rnn = self.RNN(tf.concat([Mv, r], axis=-1), initial_state=init_state)
        return rnn


class RewardLSTM2(LSTM):
    '''
    Same as LSTM, but gets rewards (or cues) as input.
    Same as RewardLSTM, but encodes the extra input jointly with the velocity input.
    '''

    def __init__(self, **kwargs):
        super(RewardLSTM2, self).__init__(**kwargs)

    def pre_g(self, inputs):
        v, p0, r = inputs
        l0 = self.encoder1(p0)
        m0 = self.encoder2(p0)
        init_state = (l0, m0)
        Mv = self.M(tf.concat([v, r], axis=-1))
        rnn = self.RNN(Mv, initial_state=init_state)
        return rnn


class LSTMPCDense(LSTM):
    def __init__(self,
                 options: Options,
                 place_cells: PlaceCells):
        super(LSTMPCDense, self).__init__(options=options, place_cells=place_cells)
        assert (self.place_cell_identity is True)
        self.Ng = options.Ng
        assert (self.Np == 2)
        self.pc_dense = Dense(options.num_pc_pred, name='pc_dense', activation=options.pc_activation)
        self.pc_k = options.pc_k
        if self.pc_k is not None:
            assert (self.pc_k <= options.num_pc_pred)

    def pc(self, inputs, g_mask=None, g_mask_add=None):
        g = self.g(inputs)
        g = mask_func(inp=g, mask_mult=g_mask, mask_add=g_mask_add)
        return self.pc_dense(g)

    def call(self, inputs, g_mask=None, g_mask_add=None):
        '''Predict positions'''
        pred_pcs = self.pc(inputs, g_mask=g_mask, g_mask_add=g_mask_add)
        if self.pc_k is not None:
            # get the top k maximally active place cells
            _, idxs = tf.math.top_k(pred_pcs, k=self.pc_k)
            pred_pcs = tf.gather(pred_pcs, idxs, batch_dims=-1)
        pred_pos = self.decoder(pred_pcs)

        return pred_pos

    def compute_loss(self, inputs, pc_outputs, pos, g_mask=None, g_mask_add=None):
        '''Compute loss and decoding error.
        Note: pc_outputs and pos are the same since trajectory generator will have place_cell_identity set to True.'''
        preds = self.call(inputs, g_mask=g_mask, g_mask_add=g_mask_add)
        loss = tf.reduce_mean(self.loss_fn(pc_outputs, preds))

        # # Weight regularization
        loss += self.weight_decay * tf.reduce_sum(self.RNN.weights[1] ** 2)

        # Compute decoding error
        # note: pred_pos and preds are the same since place cell identity is True
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = tf.reduce_mean(tf.sqrt(tf.reduce_sum((tf.cast(pos, dtype=pred_pos.dtype) - pred_pos) ** 2, axis=-1)))

        return loss, err


class LSTMPCRNN(LSTMPCDense):
    def __init__(self, options, place_cells):
        super(LSTMPCRNN, self).__init__(options=options, place_cells=place_cells)
        assert (options.pc_rnn_func is not None)
        self.pc_rnn_initial_state = options.pc_rnn_initial_state
        self.second_initial_state = (options.pc_rnn_func == "LSTM")
        if self.pc_rnn_initial_state:
            self.pc_rnn_encoder1 = Dense(options.num_pc_pred, name='pc_rnn_encoder1')
            if self.second_initial_state:  # LSTMs have tuple states
                self.pc_rnn_encoder2 = Dense(options.num_pc_pred, name='pc_rnn_encoder2')
        self.pc_rnn_func = getattr(tf.keras.layers, options.pc_rnn_func)
        self.pc_rnn = self.pc_rnn_func(options.num_pc_pred, return_sequences=True,
                                       activation=options.pc_activation,
                                       recurrent_initializer=options.initializer)

    def pc(self, inputs):
        v, p0 = inputs
        g = self.g(inputs)
        if self.pc_rnn_initial_state:
            s0 = self.pc_rnn_encoder1(p0)
            if self.second_initial_state:
                s1 = self.pc_rnn_encoder2(p0)
                init_state = (s0, s1)
            else:
                init_state = s0
            rnn = self.pc_rnn(g, initial_state=init_state)
        else:
            rnn = self.pc_rnn(g)
        return self.pc_dense(rnn)


class ThreeLayerRNNBase(Model):
    def __init__(self, options, place_cells):
        super(ThreeLayerRNNBase, self).__init__()
        self.Ng = options.Ng
        self.Np = options.Np
        self.place_field_loss = options.place_field_loss
        self.place_field_normalization = options.place_field_normalization
        self.sequence_length = options.sequence_length
        self.weight_decay = options.weight_decay
        self.place_cells = place_cells

        self.encoder = Dense(self.Ng, name='encoder')
        self.M = Dense(self.Ng, name='M')
        self.RNN = None
        self.dense = Dense(self.Ng, name='dense', activation=options.activation)
        self.decoder = Dense(self.Np, name='decoder')

        # Loss function
        self.loss_fn = create_loss_fn(
            self.place_field_loss,
            self.place_field_normalization)

    def pre_g(self, inputs):
        '''Compute rnn cell activations'''
        assert (self.RNN is not None)
        v, p0 = inputs
        s0 = self.encoder(p0)
        init_state = s0
        Mv = self.M(v)
        rnn = self.RNN(Mv, initial_state=init_state)
        return rnn

    def g(self, inputs):
        """Compute grid cell activations"""
        rnn = self.pre_g(inputs)
        g = self.dense(rnn)
        return g

    def dc(self, inputs, g_mask=None, g_mask_add=None):
        g = self.g(inputs)
        g = mask_func(inp=g, mask_mult=g_mask, mask_add=g_mask_add)
        return self.decoder(g)

    def call(self, inputs, g_mask=None, g_mask_add=None):
        """Predict place cell code"""
        place_preds = self.dc(inputs, g_mask=g_mask, g_mask_add=g_mask_add)

        return place_preds

    def compute_loss(self, inputs, pc_outputs, pos, g_mask=None, g_mask_add=None):
        """Compute loss and decoding error"""
        preds = self.call(inputs, g_mask=g_mask, g_mask_add=g_mask_add)
        loss = tf.reduce_mean(self.loss_fn(pc_outputs, preds))

        # # Weight regularization
        loss += self.weight_decay * tf.reduce_sum(self.RNN.weights[1] ** 2)

        # Compute decoding error
        pred_pos = tf.stop_gradient(self.place_cells.get_nearest_cell_pos(preds))
        err = tf.reduce_mean(tf.sqrt(tf.reduce_sum((tf.cast(pos, dtype=pred_pos.dtype) - pred_pos) ** 2, axis=-1)))

        return loss, err

    def log_weight_norms(self, epoch_idx: int):
        wandb_vals_to_log = {
            'input_matrix_norm': tf.reduce_sum(self.RNN.weights[0] ** 2).numpy(),
            'recurrent_matrix_norm': tf.reduce_sum(self.RNN.weights[1] ** 2).numpy(),
        }
        if len(self.RNN.weights) > 2:
            wandb_vals_to_log['bias_norm'] = tf.reduce_sum(self.RNN.weights[2] ** 2).numpy()

        wandb.log(wandb_vals_to_log, step=epoch_idx + 1)


class UGRNN(ThreeLayerRNNBase):
    def __init__(self, options, place_cells):
        super(UGRNN, self).__init__(options=options, place_cells=place_cells)
        self.RNN = RNN_wrapper(UGRNNCell(self.Ng, activation=options.activation),
                               return_sequences=True)

    def log_weight_norms(self, epoch_idx: int):
        wandb_vals_to_log = {
            'input_matrix_norm': tf.reduce_sum(self.RNN.weights[0] ** 2).numpy(),
            'recurrent_matrix_norm': tf.reduce_sum(self.RNN.weights[1] ** 2).numpy(),
        }
        wandb.log(wandb_vals_to_log, step=epoch_idx + 1)


class VanillaRNN(ThreeLayerRNNBase):
    def __init__(self, options, place_cells):
        super(VanillaRNN, self).__init__(options=options, place_cells=place_cells)
        self.RNN = SimpleRNN(self.Ng,
                             return_sequences=True,
                             activation=tf.keras.layers.Activation(options.activation),
                             recurrent_initializer=options.initializer,
                             name='RNN',
                             use_bias=False)


class GRU(ThreeLayerRNNBase):
    def __init__(self, options, place_cells):
        super(GRU, self).__init__(options=options, place_cells=place_cells)
        self.RNN = tf.keras.layers.GRU(self.Ng, return_sequences=True,
                                       activation=options.activation,
                                       recurrent_initializer=options.initializer)


class RewardThreeLayerRNNBase(ThreeLayerRNNBase):
    '''
    Gets rewards (or cues) as input.
    We don't encode the extra input passed to the underlying RNN,
    to mimic the GridCellAgent in DeepMind and the MERLIN paper.
    '''

    def __init__(self, **kwargs):
        super(RewardThreeLayerRNNBase, self).__init__(**kwargs)

    def pre_g(self, inputs):
        assert (self.RNN is not None)
        v, p0, r = inputs
        s0 = self.encoder(p0)
        init_state = s0
        Mv = self.M(v)
        # so that concatenation will work
        r = tf.cast(r, dtype=Mv.dtype)
        rnn = self.RNN(tf.concat([Mv, r], axis=-1), initial_state=init_state)
        return rnn


class RewardThreeLayerRNNBase2(ThreeLayerRNNBase):
    '''
    Gets rewards (or cues) as input.
    Same as RewardThreeLayerRNNBase, but encodes the extra input jointly with the velocity input.
    '''

    def __init__(self, **kwargs):
        super(RewardThreeLayerRNNBase2, self).__init__(**kwargs)

    def pre_g(self, inputs):
        assert (self.RNN is not None)
        v, p0, r = inputs
        s0 = self.encoder(p0)
        init_state = s0
        Mv = self.M(tf.concat([v, r], axis=-1))
        rnn = self.RNN(Mv, initial_state=init_state)
        return rnn


class RewardUGRNN2(RewardThreeLayerRNNBase2):
    def __init__(self, options, place_cells):
        super(RewardUGRNN2, self).__init__(options=options, place_cells=place_cells)

        self.RNN = RNN_wrapper(UGRNNCell(self.Ng, activation=options.activation),
                               return_sequences=True)


# #TODO: update this to match LSTMPCDense call and compute_loss methods if you do decide to use it
# class UGRNNPCDense(UGRNN):
#     def __init__(self, options, place_cells):
#         super(UGRNNPCDense, self).__init__(options=options, place_cells=place_cells)
#         assert(self.place_cell_identity is True)
#         self.Ng = options.Ng
#         assert(self.Np == 2)
#         self.pc_dense = Dense(options.num_pc_pred, name='pc_dense', activation=options.pc_activation)

#     def pc(self, inputs):
#         return self.pc_dense(self.g(inputs))

#     def call(self, inputs):
#         '''Predict place cell code'''
#         place_preds = self.decoder(self.pc(inputs))

#         return place_preds

class BaninoRNN(Model):
    def __init__(self, options, place_cells):
        super(BaninoRNN, self).__init__()
        '''Model definition taken from: https://github.com/deepmind/grid-cells/blob/d1a2304d9a54e5ead676af577a38d4d87aa73041/model.py'''
        self.rnn_nunits = options.banino_rnn_nunits
        self.dropout_rate = options.banino_dropout_rate
        self.Ng = options.Ng
        self.Np = options.Np
        self.Nhdc = None
        if hasattr(options, "Nhdc"):
            self.Nhdc = options.Nhdc
        self.rnn_type = options.banino_rnn_type
        self.place_field_loss = options.place_field_loss
        self.place_field_normalization = options.place_field_normalization
        self.sequence_length = options.sequence_length
        self.weight_decay = options.weight_decay
        self.place_cells = place_cells

        self.encoder1 = Dense(self.rnn_nunits,
                              kernel_initializer=initializers.VarianceScaling(scale=1.0, mode="fan_in",
                                                                              distribution="truncated_normal"),
                              bias_initializer="zeros",
                              name='encoder1')
        self._need_second_encoder = False
        if self.rnn_type.lower() == "lstm":
            self._need_second_encoder = True
            self.RNN = tf.keras.layers.LSTM(self.rnn_nunits, return_sequences=True,
                                            activation=options.activation,
                                            kernel_initializer=initializers.VarianceScaling(scale=1.0, mode="fan_in",
                                                                                            distribution="truncated_normal"),
                                            recurrent_initializer=initializers.VarianceScaling(scale=1.0, mode="fan_in",
                                                                                               distribution="truncated_normal"),
                                            bias_initializer="zeros",
                                            unit_forget_bias=True)
        else:
            raise ValueError
        if self._need_second_encoder:
            self.encoder2 = Dense(self.rnn_nunits,
                                  kernel_initializer=initializers.VarianceScaling(scale=1.0, mode="fan_in",
                                                                                  distribution="truncated_normal"),
                                  bias_initializer="zeros",
                                  name='encoder2')
        self.bottleneck_dense = Dense(self.Ng,
                                      kernel_initializer=initializers.VarianceScaling(scale=1.0, mode="fan_in",
                                                                                      distribution="truncated_normal"),
                                      bias_initializer="zeros",
                                      use_bias=False,
                                      name='dense')
        self.pc_decoder = Dense(self.Np,
                                kernel_initializer=initializers.VarianceScaling(scale=1.0, mode="fan_in",
                                                                                distribution="truncated_normal"),
                                bias_initializer="zeros",
                                name='pc_decoder')
        if self.Nhdc is not None:
            self.hdc_decoder = Dense(self.Nhdc,
                                     kernel_initializer=initializers.VarianceScaling(scale=1.0, mode="fan_in",
                                                                                     distribution="truncated_normal"),
                                     bias_initializer="zeros",
                                     name='hdc_decoder')
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate, name='dropout')
        self.loss_fn = create_loss_fn(
            self.place_field_loss,
            self.place_field_normalization)

    def pre_g(self, inputs):
        '''Compute rnn cell activations'''
        v, p0 = inputs
        l0 = self.encoder1(p0)
        if self._need_second_encoder:
            m0 = self.encoder2(p0)
            init_state = (l0, m0)
        else:
            init_state = l0
        rnn = self.RNN(v, initial_state=init_state)
        return rnn

    def g(self, inputs):
        '''Compute grid cell activations'''
        rnn = self.pre_g(inputs)
        g = self.bottleneck_dense(rnn)
        g = self.dropout(g)
        return g

    def dc(self, inputs, g_mask=None, g_mask_add=None):
        g = self.g(inputs)
        g = mask_func(inp=g, mask_mult=g_mask, mask_add=g_mask_add)
        if hasattr(self, "hdc_decoder"):
            return self.pc_decoder(g), self.hdc_decoder(g)
        else:
            return self.pc_decoder(g)

    def call(self, inputs,
             g_mask=None, g_mask_add=None,
             dc_mask=None, dc_mask_add=None,
             hdc_mask=None, hdc_mask_add=None):
        '''Predict place cell (and/or head direction) code'''
        if hasattr(self, "hdc_decoder"):
            place_preds, hdc_preds = self.dc(inputs, g_mask=g_mask, g_mask_add=g_mask_add)
            place_preds = mask_func(inp=place_preds, mask_mult=dc_mask, mask_add=dc_mask_add)
            hdc_preds = mask_func(inp=hdc_preds, mask_mult=hdc_mask, mask_add=hdc_mask_add)
            return place_preds, hdc_preds
        else:
            place_preds = self.dc(inputs, g_mask=g_mask, g_mask_add=g_mask_add)
            place_preds = mask_func(inp=place_preds, mask_mult=dc_mask, mask_add=dc_mask_add)
            return place_preds

    def compute_loss(self, inputs, cell_outputs, pos,
                     g_mask=None, g_mask_add=None,
                     dc_mask=None, dc_mask_add=None,
                     hdc_mask=None, hdc_mask_add=None):
        '''Compute loss and decoding error'''
        preds = self.call(inputs,
                          g_mask=g_mask, g_mask_add=g_mask_add,
                          dc_mask=dc_mask, dc_mask_add=dc_mask_add,
                          hdc_mask=hdc_mask, hdc_mask_add=hdc_mask_add)
        if hasattr(self, "hdc_decoder"):
            pc_outputs = cell_outputs["place_outputs"]
            hdc_outputs = cell_outputs["hdc_outputs"]
            pc_preds, hdc_preds = preds
            loss = tf.reduce_mean(
                self.loss_fn(pc_outputs, pc_preds) + tf.nn.softmax_cross_entropy_with_logits(labels=hdc_outputs,
                                                                                             logits=hdc_preds))
            loss += self.weight_decay * tf.reduce_sum(tf.square(self.hdc_decoder.weights[0]))
            pred_pos = self.place_cells.get_nearest_cell_pos(pc_preds)
        else:
            loss = tf.reduce_mean(self.loss_fn(cell_outputs, preds))
            pred_pos = self.place_cells.get_nearest_cell_pos(preds)

        # Weight regularization on weights only, not biases
        # as defined here: https://github.com/deepmind/sonnet/blob/c87468e618ed19c3720f1caa1fa8c6884b2ed36d/sonnet/src/regularizers.py#L73#L104
        loss += self.weight_decay * tf.reduce_sum(tf.square(self.bottleneck_dense.weights[0]))
        loss += self.weight_decay * tf.reduce_sum(tf.square(self.pc_decoder.weights[0]))

        # Compute decoding error
        err = tf.reduce_mean(tf.sqrt(tf.reduce_sum((tf.cast(pos, dtype=pred_pos.dtype) - pred_pos) ** 2, axis=-1)))

        return loss, err
