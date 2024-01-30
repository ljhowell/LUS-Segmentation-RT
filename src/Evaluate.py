"""
This module contains functions for evaluating the performance of segmentation models.

Functions:
    - time_performance(times) -> dict: 
        Get summary metrics from a cumulative time series
    - metric_performance(gt_masks, test_masks, metric='dice', classes=None) -> dict: 
        Calculate the metric scores between predicted and ground truth masks

"""

import numpy as np
import tensorflow as tf

from . import Metrics as metrics


def time_performance(times) -> dict:
    """
    Get summary metrics from a cumulative time series

    Args:
        times (array): Array of cumulative times per epoch [[0:t0],[1:t1],[2:t2]]

    Returns:
        dict: A summary dictionary of metrics including for timing
    """

    deltas = np.diff(np.array(times)[:, 1])
    # insert first timestep so len(deltas) == len(times_cumulative)
    deltas = np.insert(deltas, 0, np.array(times)[0, 1])

    d_times = {
        'n_epochs': len(np.array(times)),
        'time_total': np.sum(deltas),
        'av_time_per_epoch': np.mean(deltas),
        'av_time_per_epoch_excluding_first': np.mean(deltas[1:]),
        'max_time_per_epoch': np.max(deltas),
        'min_time_per_epoch': np.min(deltas),
        'stdev_time_per_epoch': np.std(deltas),
        'stdev_time_per_epoch_excluding_first': np.std(deltas[1:]),
    }

    return d_times


def metric_performance(gt_masks, test_masks, metric='dice', classes=None):
    """
    This function calculates the metric score between predicted and ground truth masks.

    Args:
        gt_masks (np.array): Ground truth masks
        test_masks (np.array): Predicted masks
        metric (str, optional): Metric to calculate from Metrics. Defaults to 'dice'.
        classes (list, optional): Classes used in training. Defaults to [0,1,2,3].

    Returns:
        dict: A dictionary showing the per class scores for the selected metric between the predicted and ground truth datasets
    """    
    if classes is None:
        classes = [0, 1, 2, 3, 4, 5]
        
    metric_fn = metrics.metrics_map(metric)  # map string to metric function

    # Convert from 1-hot to single channel
    gt_masks_argmax = np.argmax(gt_masks, axis=3)
    test_masks_argmax = np.argmax(test_masks, axis=3)

    d_scores = {c: [] for c in classes}

    for im_no in range(len(gt_masks_argmax)):
        for c in classes:
            # Get binary mask for each class
            a = tf.convert_to_tensor(
                (gt_masks_argmax[im_no] == c).astype(np.int32), dtype=tf.float32)
            b = tf.convert_to_tensor(
                (test_masks_argmax[im_no] == c).astype(np.int32), dtype=tf.float32)

            # append scores for each image
            d_scores[c].append(metric_fn(a, b).numpy())

    d_av_scores = {}
    for i in classes:
        d_av_scores[f'Average_{metric}_class_{i}'] = float(np.mean(d_scores[i]))

    d_av_scores[f'Mean_average_{metric}'] = float(
        np.array(list(d_scores.values())).flatten().mean())

    return d_av_scores
