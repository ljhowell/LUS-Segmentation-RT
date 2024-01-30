"""
Postprocessing module for semantic segmentation masks.

This module provides functions for filtering small objects in a semantic segmentation mask.

Functions:
- filter_small_objects: Filters small objects in a mask based on a minimum size threshold.

"""

from skimage.morphology import remove_small_objects, remove_small_holes
import numpy as np


def filter_small_objects(mask: np.ndarray,
                         minsize: int = 64) -> np.ndarray:
    """
    Filters small objects in a mask based on a minimum size threshold.

    Args:
        mask (ndarray): The input mask.
        minsize (int): The minimum size threshold for filtering small objects. Default is 64.

    Returns:
        ndarray: The filtered mask.

    """
    one_hot_map = []
    for class_ in [0, 1, 2, 3]:
        class_map = mask == class_
        class_map = remove_small_objects(class_map, min_size=minsize)
        class_map = remove_small_holes(class_map, area_threshold=minsize)

        one_hot_map.append(class_map)

    return np.argmax(one_hot_map, axis=0)
