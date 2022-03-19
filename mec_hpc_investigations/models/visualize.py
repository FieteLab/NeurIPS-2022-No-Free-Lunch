# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import scipy
import scipy.stats
from imageio import imsave
from tqdm.autonotebook import tqdm

from mec_hpc_investigations.models.helper_classes import PlaceCells, Options
from mec_hpc_investigations.models.trajectory_generator import TrajectoryGenerator


def concat_images(images,
                  image_width,
                  spacer_size):
    """ Concat image horizontally with spacer """
    spacer = np.ones([image_width, spacer_size, 4], dtype=np.uint8) * 255
    images_with_spacers = []

    image_size = len(images)

    for i in range(image_size):
        images_with_spacers.append(images[i])
        if i != image_size - 1:
            # Add spacer
            images_with_spacers.append(spacer)
    ret = np.hstack(images_with_spacers)
    return ret


def concat_images_in_rows(images, row_size, image_width, spacer_size=4):
    """ Concat images in rows """
    column_size = len(images) // row_size
    spacer_h = np.ones([spacer_size, image_width * column_size + (column_size - 1) * spacer_size, 4],
                       dtype=np.uint8) * 255

    row_images_with_spacers = []

    for row in range(row_size):
        row_images = images[column_size * row:column_size * row + column_size]
        row_concated_images = concat_images(row_images, image_width, spacer_size)
        row_images_with_spacers.append(row_concated_images)

        if row != row_size - 1:
            row_images_with_spacers.append(spacer_h)

    ret = np.vstack(row_images_with_spacers)
    return ret


def convert_to_colormap(im, cmap):
    im = cmap(im)
    im = np.uint8(im * 255)
    return im


def rgb(im, cmap='jet', smooth=True):
    import cv2
    cmap = plt.cm.get_cmap(cmap)
    np.seterr(invalid='ignore')  # ignore divide by zero err
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    if smooth:
        im = cv2.GaussianBlur(im, (3, 3), sigmaX=1, sigmaY=0)
    im = cmap(im)
    im = np.uint8(im * 255)
    return im


def plot_ratemaps(activations, n_plots, cmap='jet', smooth=True, width=16):
    images = [rgb(im, cmap, smooth) for im in activations[:n_plots]]
    rm_fig = concat_images_in_rows(images, n_plots // width, activations.shape[-1])
    return rm_fig


def compute_ratemaps(model,
                     options: Options,
                     res: int = 20, n_avg=None, idxs=None, trajectory_seed=0, return_all=False):
    '''Compute spatial firing fields'''

    if return_all:
        assert (hasattr(model, "g"))
        assert (hasattr(model, "dc"))
        if hasattr(model, "pre_g"):
            pre_g = np.zeros([n_avg, options.batch_size * options.sequence_length, options.Ng])
        dc = np.zeros([n_avg, options.batch_size * options.sequence_length, options.Np])

    trajectory_generator = TrajectoryGenerator(options, PlaceCells(options), trajectory_seed=trajectory_seed)

    if not n_avg:
        n_avg = 1000 // options.sequence_length

    if not np.any(idxs):
        idxs = np.arange(options.Ng)
    idxs = idxs[:options.Ng]

    g = np.zeros([n_avg, options.batch_size * options.sequence_length, options.Ng])
    if options.place_cell_predict:
        assert (hasattr(model, "pc"))
        p = np.zeros([n_avg, options.batch_size * options.sequence_length, options.num_pc_pred])
    else:
        p = np.zeros([n_avg, options.batch_size * options.sequence_length, options.Np])
    pos = np.zeros([n_avg, options.batch_size * options.sequence_length, 2])
    inp = []

    activations = np.zeros([options.Ng, res, res])
    counts = np.zeros([res, res])

    for index in tqdm(range(n_avg), leave=False, desc='Computing ratemaps'):
        inputs, pos_batch, p_batch = trajectory_generator.get_batch()
        # the element of the input tuple is velocity or cues depending on the options passed in
        relevant_inp = inputs[0]
        # batch x sequence length x dimensionality
        assert (len(relevant_inp.shape) == 3)
        inp_dim = relevant_inp.shape[-1]
        # batch * sequence length x dimensionality
        inp.append(np.reshape(relevant_inp, (-1, inp_dim)))
        if hasattr(model, "g"):  # means it is an RNN (not NMF)
            if return_all:
                if hasattr(model, "pre_g"):
                    pre_g_batch = model.pre_g(inputs)
                    assert (pre_g_batch.shape[-1] == options.Ng)
                    pre_g_batch = np.reshape(tf.gather(pre_g_batch, idxs, axis=-1), (-1, options.Ng))
                    pre_g[index] = pre_g_batch

                g_batch = model.g(inputs)
                assert (g_batch.shape[-1] == options.Ng)
                g_batch = np.reshape(tf.gather(g_batch, idxs, axis=-1), (-1, options.Ng))
                g[index] = g_batch

                dc_batch = model.dc(inputs)
                assert (dc_batch.shape[-1] == options.Np)
                dc_batch = np.reshape(model.dc(inputs), (-1, options.Np))
                dc[index] = dc_batch
            else:
                g_batch = model.g(inputs)
                assert (g_batch.shape[-1] == options.Ng)
                g_batch = np.reshape(tf.gather(g_batch, idxs, axis=-1), (-1, options.Ng))
                g[index] = g_batch

        if options.place_cell_predict:
            p_batch = np.reshape(model.pc(inputs), (-1, options.num_pc_pred))
        else:
            p_batch = np.reshape(p_batch, (-1, options.Np))
        p[index] = p_batch

        pos_batch = np.reshape(pos_batch, (-1, 2))
        pos[index] = pos_batch

        x_batch = (pos_batch[:, 0] - options.min_x) / (options.max_x - options.min_x) * res
        y_batch = (pos_batch[:, 1] - options.min_y) / (options.max_y - options.min_y) * res

        for i in range(options.batch_size * options.sequence_length):
            x = x_batch[i]
            y = y_batch[i]
            if x >= 0 and x <= res and y >= 0 and y <= res:
                counts[int(x), int(y)] += 1
                if hasattr(model, "g"):
                    activations[:, int(x), int(y)] += g_batch[i, :]

    for x in range(res):
        for y in range(res):
            if counts[x, y] > 0:
                activations[:, x, y] /= counts[x, y]

    if return_all:
        g = g.reshape((-1, options.Ng))
        dc = dc.reshape((-1, options.Np))
        g_dict = {"g": g, "dc": dc}
        if hasattr(model, "pre_g"):
            pre_g = pre_g.reshape((-1, options.Ng))
            g_dict["pre_g"] = pre_g
    else:
        g = g.reshape((-1, options.Ng))
        g_dict = g
    if options.place_cell_predict:
        p = p.reshape((-1, options.num_pc_pred))
    else:
        p = p.reshape((-1, options.Np))
    pos = pos.reshape((-1, 2))
    inp = np.stack(inp, axis=0)
    inp = inp.reshape((-1, inp.shape[-1]))
    # # scipy binned_statistic_2d is slightly slower
    # activations = scipy.stats.binned_statistic_2d(pos[:,0], pos[:,1], g.T, bins=res)[0]
    rate_map = activations.reshape((options.Ng, -1))

    return activations, rate_map, g_dict, p, pos, inp


def save_ratemaps(model, options, step, res=20, n_avg=None):
    if not n_avg:
        n_avg = 1000 // options.sequence_length
    activations, rate_map, g, p, pos, inp = compute_ratemaps(model, options, res=res, n_avg=n_avg)
    rm_fig = plot_ratemaps(activations, n_plots=len(activations))
    imdir = options.save_dir + "/" + options.run_ID
    imsave(imdir + "/" + str(step) + ".png", rm_fig)


def save_autocorr(sess, model, save_name, trajectory_generator, step, flags):
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    coord_range = ((-1.1, 1.1), (-1.1, 1.1))
    masks_parameters = zip(starts, ends.tolist())
    latest_epoch_scorer = scores.GridScorer(nbins=20, mask_parameters=masks_parameters, coords_range=coord_range)

    res = dict()
    index_size = 100
    for _ in range(index_size):
        feed_dict = trajectory_generator.feed_dict(flags.box_width_in_m, flags.box_height_in_m)
        mb_res = sess.run({
            'pos_xy': model.target_pos,
            'bottleneck': model.g,
        }, feed_dict=feed_dict)
        res = utils.concat_dict(res, mb_res)

    filename = save_name + '/autocorrs_' + str(step) + '.pdf'
    imdir = flags.save_dir + '/'
    out = utils.get_scores_and_plot(
        latest_epoch_scorer, res['pos_xy'], res['bottleneck'],
        imdir, filename)

