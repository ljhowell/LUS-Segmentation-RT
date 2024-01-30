"""
Module for calculating metrics relevant to semantic segmentation
"""
import time
import warnings
from typing import List, Dict, Union

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.math import confusion_matrix
from tensorflow import Tensor


def metrics_map(str_metric: str) -> Union[None, callable]:
    """
    Maps a string metric to its corresponding function.

    Args:
        str_metric: The string representation of the metric.

    Returns:
        The corresponding metric function if found, None otherwise.
    """
    d_map = {'dice': dice_coef,
           'iou': IoU,
           'tversky': tversky_index,
           }
    try:
        score = d_map[str_metric]
    except KeyError:
        warnings.warn(f"Unknown metric '{str_metric}', choose from {list(d_map.keys())}")
        score = None

    return score


def dice_coef(y_true: Tensor, y_pred: Tensor, smooth: float = 0.0) -> Tensor:
    """
    Computes the Dice coefficient between the true and predicted masks.

    Args:
        y_true: The true mask.
        y_pred: The predicted mask.
        smooth: Smoothing factor.

    Returns:
        The Dice coefficient.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)  # tp
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def IoU(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Computes the Intersection over Union (IoU) between the true and predicted masks.

    Args:
        y_true: The true mask.
        y_pred: The predicted mask.

    Returns:
        The IoU score.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return intersection/union


def precision(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Computes the precision between the true and predicted masks.

    Args:
        y_true: The true mask.
        y_pred: The predicted mask.

    Returns:
        The precision score.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    tp = K.sum(y_true_f * y_pred_f)
    fp = K.sum((1-y_true_f) * y_pred_f)
    return tp / (tp + fp)


def recall(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Computes the recall between the true and predicted masks.

    Args:
        y_true: The true mask.
        y_pred: The predicted mask.

    Returns:
        The recall score.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    tp = K.sum(y_true_f * y_pred_f)
    fn = K.sum(y_true_f * (1-y_pred_f))
    return tp/(tp + fn)


def tversky_index(y_true: Tensor, 
                  y_pred: Tensor, 
                  alpha: float = 0.7, 
                  beta: float = 0.3, 
                  smooth: float = 0.0) -> Tensor:
    """
    Computes the Tversky index between the true and predicted masks.

    Args:
        y_true: The true mask.
        y_pred: The predicted mask.
        alpha: Weight for false negatives.
        beta: Weight for false positives.
        smooth: Smoothing factor.

    Returns:
        The Tversky index.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    tp = K.sum(y_true_f * y_pred_f)
    fn = K.sum(y_true_f * (1-y_pred_f))
    fp = K.sum((1-y_true_f) * y_pred_f)
    return (tp + smooth) / (tp + alpha*fn + beta*fp + smooth)


def conf_matrix(y_true: Tensor, 
                y_pred: Tensor, 
                num_classes: int = None) -> Tensor:
    """
    Computes the confusion matrix between the true and predicted masks.

    Args:
        y_true: The true mask.
        y_pred: The predicted mask.
        num_classes: The number of classes.

    Returns:
        The confusion matrix.
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    cm = confusion_matrix(y_true_f, y_pred_f, num_classes=num_classes)
    return cm


def metric_average(y_true: Tensor, 
                   y_pred: Tensor, 
                   metric_fn: callable, 
                   num_classes: int = None, 
                   class_weights: List[float] = None) -> float:
    """
    Computes the average metric score across all classes.

    Args:
        y_true: The true mask.
        y_pred: The predicted mask.
        metric_fn: The metric function.
        num_classes: The number of classes.
        class_weights: The weights for each class.

    Returns:
        The average metric score.
    """
    if num_classes == None:
        # assume the number of classes == number of channels in the mask (one-hot encoded)
        num_classes = y_true.shape[3]
    if class_weights == None:
        class_weights = [1 for i in range(num_classes)]

    metric = 0
    for i in range(num_classes):
        metric += metric_fn(y_true[:, :, :, i],
                            y_pred[:, :, :, i]) * class_weights[i]
    return metric/num_classes


def per_class_scores(y_true: Tensor, 
                     y_pred: Tensor, 
                     metric_fn: callable = dice_coef, 
                     num_classes: int = None) -> Dict[Union[int, str], float]:
    """
    Computes the metric scores for each class.

    Args:
        y_true: The true mask.
        y_pred: The predicted mask.
        metric_fn: The metric function.
        num_classes: The number of classes.

    Returns:
        A dictionary containing the metric scores for each class and the average score.
    """
    if num_classes is None:
        # assume the number of classes == number of channels in the mask (one-hot encoded)
        num_classes = y_true.shape[3]

    d_scores = {i: None for i in range(num_classes)}

    y_pred = K.argmax(y_pred, axis=3)  # one-hot decode
    for i in range(num_classes):
        # get only pixels predicted for class
        y_pred_class = K.cast(K.equal(y_pred, i), 'float32')
        d_scores[i] = metric_fn(y_true[:, :, :, i],  y_pred_class).numpy()

    d_scores['average'] = sum(d_scores.values()) / num_classes

    return d_scores


def precision_average(y_true: Tensor, y_pred: Tensor) -> float:
    """
    Computes the average precision score across all classes.

    Args:
        y_true: The true mask.
        y_pred: The predicted mask.

    Returns:
        The average precision score.
    """
    return metric_average(y_true, y_pred, precision)


def recall_average(y_true: Tensor, y_pred: Tensor) -> float:
    """
    Computes the average recall score across all classes.

    Args:
        y_true: The true mask.
        y_pred: The predicted mask.

    Returns:
        The average recall score.
    """
    return metric_average(y_true, y_pred, recall)


def dice_coef_average(y_true: Tensor, y_pred: Tensor) -> float:
    """
    Computes the average Dice coefficient score across all classes.

    Args:
        y_true: The true mask.
        y_pred: The predicted mask.

    Returns:
        The average Dice coefficient score.
    """
    return metric_average(y_true, y_pred, dice_coef)


def IoU_average(y_true: Tensor, y_pred: Tensor) -> float:
    """
    Computes the average IoU score across all classes.

    Args:
        y_true: The true mask.
        y_pred: The predicted mask.

    Returns:
        The average IoU score.
    """
    return metric_average(y_true, y_pred, IoU)


def tversky_average(y_true: Tensor, y_pred: Tensor) -> float:
    """
    Computes the average Tversky index score across all classes.

    Args:
        y_true: The true mask.
        y_pred: The predicted mask.

    Returns:
        The average Tversky index score.
    """
    return metric_average(y_true, y_pred, tversky_index)


class timecallback(Callback):
    def __init__(self):
        self.times = []
        # use this value as reference to calculate cummulative time taken
        self.timetaken = time.perf_counter()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append((epoch, time.perf_counter() - self.timetaken))
