""" Loss functions for training segmentation models"""
import warnings

from tensorflow.keras.losses import categorical_crossentropy
import tensorflow.keras.backend as K

from . import Metrics as metrics


def loss_functions_map(str_loss: str):
    """Map loss function name to its corresponding function

    Args:
        str_loss (str): Key for the loss function. If 'print', display available keys.

    Returns:
        Loss function: pointer to loss function
    """

    d_map = {'dice': dice_loss,
             'iou': IoU_loss,
             'tversky': tversky_loss,
             'focal_tversky': focal_tversky_loss,
             'cross_entropy': cat_crossentropy,
             'combined_ce_dice': combined_ce_dice_loss,
             'combined_ce_focal_tversky': combined_ce_focal_tversky_loss,
             }

    if str_loss == 'print':
        print('Available loss functions:', list(d_map.keys()))
        return None

    else:
        try:
            loss_fn = d_map[str_loss]
        except KeyError:
            warnings.warn(
                f"Unknown loss function '{str_loss}', choose from {list(d_map.keys())}")
            loss_fn = None

        return loss_fn


def dice_loss(y_true, y_pred):
    """Dice loss"""
    return 1-metrics.dice_coef_average(y_true, y_pred)


def IoU_loss(y_true, y_pred):
    """IoU Loss"""
    return 1-metrics.IoU_average(y_true, y_pred)


def tversky_loss(y_true, y_pred):
    "Tversky Loss"
    return 1-metrics.tversky_average(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred, gamma=3/4):
    """Focal Tversky Loss"""
    return K.pow((1-metrics.tversky_average(y_true, y_pred)), gamma)


def combined_ce_dice_loss(y_true, y_pred, mixing_factor=1/3):
    """Combo loss of cross entropy and dice losses"""
    return mixing_factor * categorical_crossentropy(y_true, y_pred) + (1-mixing_factor)*dice_loss(y_true, y_pred)


def combined_ce_focal_tversky_loss(y_true, y_pred, mixing_factor=1/3):
    """Combo loss of cross entropy and tversky losses"""
    return mixing_factor * categorical_crossentropy(y_true, y_pred) + (1-mixing_factor)*focal_tversky_loss(y_true, y_pred)


def cat_crossentropy(y_true, y_pred):
    """cross entropy loss"""
    return categorical_crossentropy(y_true, y_pred)
