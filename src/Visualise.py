"""
This module provides functions for visualizing images and masks in semantic segmentation tasks.

Functions:
- cv2_imshow(im: np.array, save: str = '') -> None: 
    Helper function to show a cv2 image (in BGR format)
- seg_to_rgba(mask: np.array, cm: tuple = plt.cm.tab10.colors) -> np.array: 
    Convert a semantic segmentation mask to RGBA using a prescribed colour map
- seg_to_rgb(mask: np.array, cm: tuple = plt.cm.tab10.colors) -> np.array: 
    Convert a semantic segmentation mask to RGB using a prescribed colour map
- one_hot_decode(mask: np.array) -> np.array: 
    Decode a one-hot encoded mask to a single-channel mask
- display_image_mask(image: np.array = np.zeros(1), 
                            mask: np.array = np.zeros(1), 
                            axs = None, 
                            d_classes: dict = None, 
                            alpha: float = 0.5, 
                            mask_colours: tuple = None, 
                            cmap: str = 'gray', 
                            save: str = None, 
                            title: str = None, 
                            figsize: list = (10,10), 
                            legend_loc: str = 'best', 
                            font_size: int = 15, 
                            show: bool = True) -> None: 
        Display an image and overlapping mask where each class is shown in a different colour
- create_cv2_legend(classes: list, cmap: tuple = None) -> np.array: Create a legend for a cv2 image
"""
import numpy as np
from matplotlib import pyplot as plt
import cv2


def cv2_imshow(im: np.array, save: str = '') -> None:
    """Helper function to show a cv2 image (in BGR format)

    Args:
        im (np.array): Image as read-in by cv2
        save (str, optional): Save the image. Defaults to ''.
    """
    im = im[..., ::-1]  # Convert BGR to RGB

    plt.figure(figsize=(14, 10))
    plt.axis('off')
    plt.imshow(im)

    if save != '':
        plt.savefig(save)


def seg_to_rgba(mask: np.array, cm: tuple = plt.cm.tab10.colors) -> np.array:
    """Convert a semantic segmanetation mask to RGBA using a prescribed colour map

    Args:
        mask (np.array): 1-channel image, 0 is background, 1,2,... are different classes. 
        cm (tuple, optional): Colours to use. Defaults to plt.cm.tab10.colors.

    Returns:
        np.array: RGBA image showing classes as different colours
    """
    h, w = mask.shape
    img_rgba = np.zeros((h, w, 4))

    lut = {label+1: list(rgb_colour)+[1]
           for label, rgb_colour in enumerate(cm)}
    lut[0] = [0, 0, 0, 0]

    for label, rgb_colour in lut.items():
        img_rgba[mask == label] = rgb_colour

    # if img_rgba.sum() == 0:
    #     print('WARNING: No classes found in mask, check pixels have values 0,1,2,... ')
    return img_rgba


def seg_to_rgb(mask: np.array, cm: tuple = plt.cm.tab10.colors) -> np.array:
    """Convert a semantic segmanetation mask to RGB using a prescribed colour map

    Args:
        mask (np.array): 1-channel image, 0 is background, 1,2,... are different classes. 
        cm (tuple, optional): Colours to use. Defaults to plt.cm.tab10.colors.

    Returns:
        np.array: RGB image showing classes as different colours
    """
    h, w = mask.shape
    img_rgb = np.zeros((h, w, 3))

    lut = {label+1: list(rgb_colour) for label, rgb_colour in enumerate(cm)}
    lut[0] = [0, 0, 0]

    for label, rgb_colour in lut.items():
        img_rgb[mask == label] = rgb_colour

    # if img_rgb.sum() == 0:
    #     print('WARNING: No classes found in mask, check pixels have values 0,1,2,... ')
    return img_rgb


def one_hot_decode(mask: np.array) -> np.array:
    """
    One-hot decode a mask

    Args:
        mask (np.array): Mask to decode

    Returns:
        np.array: Decoded mask
    """
    return mask.sum(axis=2)


def display_image_mask(image: np.array = np.zeros(1),
                       mask: np.array = np.zeros(1),
                       axs=None,
                       d_classes: dict = None,
                       alpha: float = 0.5,
                       mask_colours: tuple = None,
                       cmap: str = 'gray',
                       save: str = None,
                       title: str = None,
                       figsize: list = (10, 10),
                       legend_loc: str = 'best',
                       font_size: int = 15,
                       show: bool = True) -> None:
    """Display an image and overlapping mask where each class is shown in a different colour

    Args:
        image (np.array): Image
        mask (np.array): Mask
        axs (matplotlib.axis): Axis to plot or None
        d_classes (dict, optional): Mapping from int to obj classes {1:'Class 1', 2:'Class 2'}. 
                                    Defaults to None.
        alpha (float, optional): Mask opacity. Defaults to 0.5.
        mask_colours (tuple, optional): Colours to use for segmentation masks. 
                                        Defaults to plt.cm.tab10.colors.
        cmap (str, optional): Colour map to use for image. Defaults to 'gray.
        save (str, optional): Save the image (path),
        title (str, optional): Plot title
        figsize (list, optional): Figure size to plot
        leged_locs (list, optional): Location of the legend
        font_size (int, optional): Font size for legend
    """
    if mask_colours is None:
        mask_colours = list(plt.cm.tab10.colors)
        # remove grey and brown as these have poor contrast on US images
        mask_colours.pop(5)
        mask_colours.pop(6)

    if d_classes is None:
        d_classes = {i+1: f'Class {i}' for i in range(9)}

    if mask is None:
        mask = np.zeros(1)

    d_colours = {d_classes[i]: mask_colours[i-1]
                 for i in d_classes.keys()}  # map colours to classes

    # Plotting
    if axs is None:
        _, axs = plt.subplots(figsize=figsize)

    axs.axis('off')

    # Show image
    if image.any():
        ax = axs.imshow(image[:, :, ::-1], cmap=cmap, interpolation='none')
        ax.set_clim(0, 1)

    # Show mask (optional)
    if mask.any():
        if len(mask.shape) == 3:  # h,w,c
            if mask.shape[2] == 3:  # c=3
                mask = one_hot_decode(mask)

        axs.imshow(seg_to_rgba(mask, cm=d_colours.values()),
                   alpha=alpha, interpolation='none')

        # Legend

        if legend_loc is not None:
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', label=cla,
                           markerfacecolor=d_colours[cla], markersize=15)
                for i, cla in d_classes.items()
                if i in np.unique(mask)
            ]

            if legend_loc == 'above':
                legend_loc = 'upper center'
                bbox_to_anchor = (0., 1.02, 1., .102)
                ncol = 3
            elif legend_loc == 'below':
                legend_loc = 'lower center'
                bbox_to_anchor = (0., -0.1, 1., .102)
                ncol = 3

            else:
                bbox_to_anchor = None
                ncol = 1

            axs.legend(handles=legend_elements, loc=legend_loc,
                       bbox_to_anchor=bbox_to_anchor, ncol=ncol, prop={'size': font_size})

    if title:
        axs.set_title(title, fontsize=22)

    if save is not None:
        plt.savefig(save, bbox_inches='tight', pad_inches=0, dpi=300)

    if show is False:
        plt.close()


def create_cv2_legend(classes: list,
                      cmap: tuple = None) -> np.array:
    """
    Create a legend for a cv2 image

    Args:
        classes (list): List of classes
        cmap (tuple): Colours to use for the legend

    Returns:
        np.array: Legend image
    """
    legend = np.zeros(((len(classes) * 25) + 25, 300, 3), dtype="uint8")

    if cmap is None:
        cmap = list(plt.cm.tab10.colors)
        # remove grey and brown as these have poor contrast on US images
        cmap.pop(5)
        cmap.pop(6)

    for i, c in enumerate(classes):
        cv2.putText(legend, c, (5, (i * 25) + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(legend, (160, (i * 25)),
                      (300, (i * 25) + 25), cmap[i][::-1], -1)
    return legend
