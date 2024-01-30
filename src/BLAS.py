import warnings
import numpy as np
from scipy.integrate import simpson

CLASS_PL = 2
CLASS_B_LINE = 4
CLASS_CONFLUENCE = 5


def roi_bbox(im: np.ndarray) -> tuple:
    """
    Calculate the bounding box coordinates of the region of interest (ROI) in the image for the BLAS calculation.

    Args:
        im (np.ndarray): The input image.

    Returns:
        tuple: The top, bottom, left, and right coordinates of the ROI.

    Raises:
        UserWarning: If no pleural line is found.
        UserWarning: If no Confluence and no b-line is found.
        UserWarning: If there is an error in the ROI.

    """

    # Get the top, bottom, left and right coordinates of the ROI
    try:
        top = np.max(np.where(im == CLASS_PL)[0])
        left = np.min(np.where(im == CLASS_PL)[1])
        right = np.max(np.where(im == CLASS_PL)[1])

    except UserWarning:
        warnings.warn('No pleural line')
        return None

    # Get the bottom of the ROI
    bottom_bline_indices = np.where(im == CLASS_B_LINE)[0]
    bottom_confluence_indices = np.where(im == CLASS_CONFLUENCE)[0]

    if len(bottom_bline_indices) == 0 and len(bottom_confluence_indices) == 0:
        warnings.warn('No Confluence or b-line(s) found in image')
        return None

    bottom_bline = np.max(bottom_bline_indices) if len(bottom_bline_indices) > 0 else 0
    bottom_confluence = np.max(bottom_confluence_indices) if len(bottom_confluence_indices) > 0 else 0

    bottom = np.max([bottom_bline, bottom_confluence]) + 1

    # Assert physical dimensions
    if top >= bottom:
        warnings.warn('ROI Error')
        return None
    if left >= right:
        warnings.warn('ROI Error')
        return None

    return (top, bottom, left, right)


def get_roi(im: np.ndarray) -> np.ndarray:
    """
    Extract the region of interest (ROI) from the image.

    Args:
        im (np.ndarray): The input image.

    Returns:
        np.ndarray: The ROI.

    """
    roi = roi_bbox(im)

    if roi is None:
        return None

    top, bottom, left, right = roi
    im = im[top:bottom, left:right]

    return im


def bline_fraction(roi: np.ndarray) -> np.ndarray:
    """
    Calculate the b-line fraction of the ROI.

    Args:
        roi (np.ndarray): The ROI.

    Returns:
        np.ndarray: The b-line fraction.

    """
    if roi is None:
        return None
    
    class_b_line = 4
    class_b_line_confluence = 5
    roi_consolidaion = np.logical_or(
        roi == class_b_line, roi == class_b_line_confluence).mean(axis=1)

    return roi_consolidaion


def calc_blas(mask: np.ndarray) -> float:
    """
    Calculate the B-line Artifact Score (BLAS) of the mask.

    Args:
        mask (np.ndarray): The input mask.

    Returns:
        float: The BLAS.

    """
    roi = get_roi(mask)
    if roi is None:
        return 0

    roi_consolidaion = bline_fraction(roi)
    # roi_consolidaion = roi_consolidaion*1.25 # Account for shadow region
    # roi_consolidaion[roi_consolidaion>1] = 1 # clip to 1

    return simpson(roi_consolidaion) / len(roi_consolidaion)
