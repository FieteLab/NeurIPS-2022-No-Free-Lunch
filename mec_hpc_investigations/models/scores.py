# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# assert(self._coords_range is not None)
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Grid score calculations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from skimage.feature import peak_local_max
from scipy.signal import correlate2d
from typing import Tuple, Union


def circle_mask(size, radius, in_val=1.0, out_val=0.0):
    """Calculating the grid scores with different radius."""
    sz = [math.floor(size[0] / 2.0), math.floor(size[1] / 2.0)]
    x = np.linspace(-sz[0], sz[1], size[1])
    x = np.expand_dims(x, 0)
    x = x.repeat(size[0], 0)
    y = np.linspace(-sz[0], sz[1], size[1])
    y = np.expand_dims(y, 1)
    y = y.repeat(size[1], 1)
    z = np.sqrt(x ** 2 + y ** 2)
    z = np.less_equal(z, radius)
    vfunc = np.vectorize(lambda b: b and in_val or out_val)
    return vfunc(z)


class GridScorer(object):
    """Class for scoring ratemaps given trajectories."""

    def __init__(self, nbins, mask_parameters, coords_range=None, min_max=True):
        """Scoring ratemaps given trajectories.
        Args:
          nbins: Number of bins per dimension in the ratemap.
          coords_range: Environment coordinates range.
          mask_parameters: parameters for the masks that analyze the angular
            autocorrelation of the 2D autocorrelation.
          min_max: Minimum difference between correlations of spatial autocorrelograms rotated at 30 and 60 degrees
                   with those at 30, 90, and 150 degrees. grid_score_60 is the value that I have seen used in the literature (e.g. Hardcastle et al. 2017).
                   Note: Butler, Hardcastle et al. 2019 do the differences of the means, so min_max=False if you want to do that.
        """
        self._nbins = nbins
        self._min_max = min_max
        self._coords_range = coords_range
        self._corr_angles = [30, 45, 60, 90, 120, 135, 150]
        # Create all masks
        self._masks = [(self._get_ring_mask(mask_min, mask_max), (mask_min,
                                                                  mask_max))
                       for mask_min, mask_max in mask_parameters]
        # Mask for hiding the parts of the SAC that are never used
        self._plotting_sac_mask = circle_mask(
            [self._nbins * 2 - 1, self._nbins * 2 - 1],
            self._nbins,
            in_val=1.0,
            out_val=np.nan)

    def calculate_ratemap(self, xs, ys, activations, statistic='mean'):
        assert (self._coords_range is not None)
        return scipy.stats.binned_statistic_2d(
            xs,
            ys,
            activations,
            bins=self._nbins,
            statistic=statistic,
            range=self._coords_range)[0]

    def _get_ring_mask(self, mask_min, mask_max):
        n_points = [self._nbins * 2 - 1, self._nbins * 2 - 1]
        return (circle_mask(n_points, mask_max * self._nbins) *
                (1 - circle_mask(n_points, mask_min * self._nbins)))

    def grid_score_60(self, corr):
        if self._min_max:
            return np.minimum(corr[60], corr[120]) - np.maximum(
                corr[30], np.maximum(corr[90], corr[150]))
        else:
            return ((corr[60] + corr[120]) / 2.0) - ((corr[30] + corr[90] + corr[150]) / 3.0)

    def grid_score_90(self, corr):
        return corr[90] - ((corr[45] + corr[135]) / 2.0)

    def calculate_sac(self, seq1):
        """Calculating spatial autocorrelogram."""
        seq2 = seq1

        def filter2(b, x):
            stencil = np.rot90(b, 2)
            return scipy.signal.convolve2d(x, stencil, mode='full')

        seq1 = np.nan_to_num(seq1)
        seq2 = np.nan_to_num(seq2)

        ones_seq1 = np.ones(seq1.shape)
        ones_seq1[np.isnan(seq1)] = 0
        ones_seq2 = np.ones(seq2.shape)
        ones_seq2[np.isnan(seq2)] = 0

        seq1[np.isnan(seq1)] = 0
        seq2[np.isnan(seq2)] = 0

        seq1_sq = np.square(seq1)
        seq2_sq = np.square(seq2)

        seq1_x_seq2 = filter2(seq1, seq2)
        sum_seq1 = filter2(seq1, ones_seq2)
        sum_seq2 = filter2(ones_seq1, seq2)
        sum_seq1_sq = filter2(seq1_sq, ones_seq2)
        sum_seq2_sq = filter2(ones_seq1, seq2_sq)
        n_bins = filter2(ones_seq1, ones_seq2)
        n_bins_sq = np.square(n_bins)

        std_seq1 = np.power(
            np.subtract(
                np.divide(sum_seq1_sq, n_bins),
                (np.divide(np.square(sum_seq1), n_bins_sq))), 0.5)
        std_seq2 = np.power(
            np.subtract(
                np.divide(sum_seq2_sq, n_bins),
                (np.divide(np.square(sum_seq2), n_bins_sq))), 0.5)
        covar = np.subtract(
            np.divide(seq1_x_seq2, n_bins),
            np.divide(np.multiply(sum_seq1, sum_seq2), n_bins_sq))
        x_coef = np.divide(covar, np.multiply(std_seq1, std_seq2))
        x_coef = np.real(x_coef)
        x_coef = np.nan_to_num(x_coef)
        return x_coef

    @staticmethod
    def calculate_grid_cell_periodicity_and_orientation(rate_map: np.ndarray,
                                                        ) -> Tuple[float, float, np.ndarray]:
        """
        Given a rate map with suspected grid cells, compute the

        :param rate_map:
        :return:
        """

        # Copied from Mikail
        autocorr = correlate2d(rate_map, rate_map, mode='full')
        # autocorr = correlate2d(smoothed_rates_wf,smoothed_rates_wf,mode = 'full')
        tmp = autocorr
        maxes = peak_local_max(tmp)
        maxes_from_center = maxes - np.array(tmp.shape) // 2
        distances = np.linalg.norm(maxes_from_center, axis=1)
        sorted_indices = np.argsort(distances)
        nearest_six = maxes_from_center[sorted_indices[1:7]]
        period, period_err = np.average(np.linalg.norm(nearest_six, axis=1)), np.std(
            np.linalg.norm(nearest_six, axis=1))
        orientations = np.arctan2(nearest_six[:, 1], nearest_six[:, 0])
        return period, period_err, orientations

    def rotated_sacs(self, sac, angles):
        return [
            scipy.ndimage.interpolation.rotate(sac, angle, reshape=False)
            for angle in angles
        ]

    def get_grid_scores_for_mask(self, sac, rotated_sacs, mask):
        """Calculate Pearson correlations of area inside mask at corr_angles."""
        masked_sac = sac * mask
        ring_area = np.sum(mask)
        # Calculate dc on the ring area
        masked_sac_mean = np.sum(masked_sac) / ring_area
        # Center the sac values inside the ring
        masked_sac_centered = (masked_sac - masked_sac_mean) * mask
        variance = np.sum(masked_sac_centered ** 2) / ring_area + 1e-5
        corrs = dict()
        for angle, rotated_sac in zip(self._corr_angles, rotated_sacs):
            masked_rotated_sac = (rotated_sac - masked_sac_mean) * mask
            cross_prod = np.sum(masked_sac_centered * masked_rotated_sac) / ring_area
            corrs[angle] = cross_prod / variance
        return self.grid_score_60(corrs), self.grid_score_90(corrs), variance

    def get_scores(self, rate_map):
        """Get summary of scrores for grid cells."""
        assert (len(rate_map.shape) == 2)
        assert (rate_map.shape[0] == rate_map.shape[1])
        assert (rate_map.shape[0] == self._nbins)

        sac = self.calculate_sac(rate_map)
        rotated_sacs = self.rotated_sacs(sac, self._corr_angles)

        scores = [
            self.get_grid_scores_for_mask(sac, rotated_sacs, mask)
            for mask, mask_params in self._masks  # pylint: disable=unused-variable
        ]
        scores_60, scores_90, variances = map(np.asarray, zip(*scores))  # pylint: disable=unused-variable
        max_60_ind = np.argmax(scores_60)
        max_90_ind = np.argmax(scores_90)

        return (scores_60[max_60_ind], scores_90[max_90_ind],
                self._masks[max_60_ind][1], self._masks[max_90_ind][1], sac, max_60_ind)

    def plot_ratemap(self, ratemap, ax=None, title=None, *args, **kwargs):  # pylint: disable=keyword-arg-before-vararg
        """Plot ratemaps."""
        if ax is None:
            ax = plt.gca()
        # Plot the ratemap
        ax.imshow(ratemap, interpolation='none', *args, **kwargs)
        # ax.pcolormesh(ratemap, *args, **kwargs)
        ax.axis('off')
        if title is not None:
            ax.set_title(title)

    def plot_sac(self,
                 sac,
                 mask_params=None,
                 ax=None,
                 title=None,
                 *args,
                 **kwargs):  # pylint: disable=keyword-arg-before-vararg
        """Plot spatial autocorrelogram."""
        if ax is None:
            ax = plt.gca()
        # Plot the sac
        useful_sac = sac * self._plotting_sac_mask
        ax.imshow(useful_sac, interpolation='none', *args, **kwargs)
        # ax.pcolormesh(useful_sac, *args, **kwargs)
        # Plot a ring for the adequate mask
        if mask_params is not None:
            center = self._nbins - 1
            ax.add_artist(
                plt.Circle(
                    (center, center),
                    mask_params[0] * self._nbins,
                    # lw=bump_size,
                    fill=False,
                    edgecolor='k'))
            ax.add_artist(
                plt.Circle(
                    (center, center),
                    mask_params[1] * self._nbins,
                    # lw=bump_size,
                    fill=False,
                    edgecolor='k'))
        ax.axis('off')
        if title is not None:
            ax.set_title(title)
