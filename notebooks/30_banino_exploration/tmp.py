import tensorflow as tf
from typing import Tuple, Union


class PlaceCells(object):

    def __init__(self,
                 Np: int,
                 place_field_loss: str = 'crossentropy',
                 place_field_values: str = 'difference_of_gaussians',
                 place_field_normalization: str = 'global',
                 n_place_fields_per_cell: Union[int, str] = 1,
                 place_cell_rf: Union[float, str] = 0.12,  # in cm
                 surround_scale: Union[float, str] = 2.,
                 min_x: float = -1.1,
                 max_x: float = 1.1,
                 min_y: float = -1.1,
                 max_y: float = 1.1,):

        self.place_field_loss = place_field_loss

        assert place_field_values in {
            'cartesian',
            'gaussian',
            'difference_of_gaussians',
            'true_difference_of_gaussians',
            'softmax_of_differences',
        }
        if place_field_values == 'cartesian':
            assert Np == 2
        self.place_field_values = place_field_values

        assert place_field_normalization in {
            'none',
            'local',
            'global',
        }
        self.place_field_normalization = place_field_normalization

        self.Np = Np
        self.place_field_loss = place_field_loss
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

        # Choose locations for place cells.
        self.n_place_fields_per_cell = n_place_fields_per_cell
        if isinstance(self.n_place_fields_per_cell, int):

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

        elif isinstance(n_place_fields_per_cell, str):

            if n_place_fields_per_cell.startswith('Poisson'):
                rate = self.extract_floats_from_str(n_place_fields_per_cell)
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

        # Create place cell receptive field tensor.
        if isinstance(place_cell_rf, (float, int)):
            # Add the 1, 1, to the shape for future broadcasting
            self.place_cell_rf = float(place_cell_rf) * tf.ones(
                shape=(1, 1, self.Np, self.max_n_place_fields_per_cell),
                dtype=tf.float32)
        elif isinstance(place_cell_rf, str):
            if place_cell_rf.startswith('Uniform'):
                low, high = self.extract_floats_from_str(s=place_cell_rf)
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
        if isinstance(surround_scale, (float, int)):
            # Add the 1, 1, to the shape for future broadcasting
            self.surround_scale = float(surround_scale) * tf.ones(
                shape=(1, 1, self.Np, self.max_n_place_fields_per_cell),
                dtype=tf.float32)
        elif isinstance(surround_scale, str):
            if surround_scale.startswith('Uniform'):
                # Add the 1, 1, to the shape for future broadcasting
                low, high = self.extract_floats_from_str(s=surround_scale)
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

        # Shape: (batch size, sequence length, num place cells, max num fields per cell, 2)
        d = tf.abs(pos[:, :, tf.newaxis, tf.newaxis, :] - self.us[tf.newaxis, tf.newaxis, ...])

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

        if self.place_field_normalization == 'local':
            normalized_dist_squared = tf.exp(-min_divided_dist_squared)
        elif self.place_field_normalization == 'global':
            normalized_dist_squared = tf.nn.softmax(-min_divided_dist_squared, axis=2)
        else:
            raise ValueError(f"Impermissible normalization: {self.place_field_normalization}")

        if self.place_field_values == 'gaussian':
            # Shape: (batch size, sequence length, num place cells,)
            outputs = normalized_dist_squared

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

            if self.place_field_normalization == 'local':
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


pc = PlaceCells(Np=512,
                place_cell_rf='Uniform( 1.5 , 2.5 )')

batch_size = 11  # doesn't matter
trajectory_length = 17  # also doesn't matter
positions_to_get_activity_at = tf.random.uniform(
                shape=(batch_size, trajectory_length, 2),
                minval=-1.1,
                maxval=1.1,
                dtype=tf.float32)

pc_activations = pc.get_activation(pos=positions_to_get_activity_at)
pc_activations_numpy = pc_activations.numpy()
print('Done!')
