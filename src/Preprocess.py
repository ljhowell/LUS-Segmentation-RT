"""
This module provides functions for preprocessing image data and 
converting VIA polygon labels to mask labels.

Functions:
- get_region_properties(region: dict) -> tuple: 
    Get the coordinates and classes of a polygon from a VIA label.
- mask_image(filename: str, ls_polygons: list, ls_classes: list, path_data: str) -> tuple: 
    Convert a polygon label to a mask label.
- create_roi_mask(image_shape, coordinates) -> np.ndarray: 
    Create a region of interest (ROI) mask from the given coordinates.
- poly_to_Mask(labels: dict, path_data: str) -> tuple: 
    Convert a VIA polygon label to a mask label.
- save_mask(filename, mask): 
    Save a mask image to a file.
- read_mask(filename) -> np.ndarray: 
    Read a mask image from a file.
- read_image(filename) -> np.ndarray: 
    Read an image from a file.
- read_json(json_path: str) -> dict: 
    Read a JSON file and return the corresponding dictionary.
- list_to_dict(ls: list) -> dict: 
    Convert a list to a dictionary with natural numbers as keys.
- dict_to_list(d) -> list: 
    Convert a dictionary to a list (drop keys).
- dict_to_json(d: dict, filename: str) -> None: 
    Save a Python dictionary to a JSON file.
- list_files(path: str, ext: str = '.jpg', include_path: bool = True) -> list: 
    Get a list of all files with a certain extension in a directory.
- via_to_mrcnn(img_dir: str, d_labels: dict, v=True, vv=False) -> dict: 
    Convert VIA labels to Mask R-CNN labels.

Note: This module requires the following dependencies: 
    logging, os, json, numpy, cv2, skimage.draw, detectron2.structures.BoxMode.
"""
import logging
import os
import json
import warnings
from typing import List, Tuple

import numpy as np
import cv2
from skimage import draw

def get_region_properties(region: dict) -> tuple:
    """Get the coords and classes of a polygon from a VIA label

    Args:
        region (dict): dict with single polygon label 

    Returns:
        tuple: (objects=[class], coordinates=[(x,y)])
    """

    shape_attributes = region["shape_attributes"]
    region_attributes = region["region_attributes"]

    if 'roadscene' not in region_attributes.keys():
        warnings.warn('Missing annotation detected in Preprocess.get_region_properties. \
                        Must contain "roadscene" attribute')

    objects = region_attributes["roadscene"]

    all_points_x = shape_attributes["all_points_x"]
    all_points_y = shape_attributes["all_points_y"]
    coordinates = []

    for i in range(0, len(all_points_x)):
        coordinates.append((all_points_x[i], all_points_y[i]))

    return (objects, coordinates)


def mask_image(filename: str,
               ls_polygons: list,
               ls_classes: list,
               path_data: str
               ):
    """Convert a polygon label to a mask label

    Args:
        filename (str): Filename of image
        ls_polygons (list): List of polygon coords in label
        ls_classes (list): List of classes 
        path_data (srt): Path to the image data

    Returns:
        image, mask: Arrays for the image and the corresponding mask
    """

    image = cv2.imread(os.path.join(path_data, filename))
    mask = np.zeros(shape=image.shape, dtype="uint8")
    i = 0
    for points in ls_polygons:
        points = [(int(x), int(y)) for x, y in points]

        points = np.array(points)

        try:
            int_class = int(ls_classes[i])
        except ValueError:
            logging.warning('Non-integer class ID detected')

        mask = cv2.drawContours(
            mask, [points], -1, (int_class, int_class, int_class), cv2.FILLED)

        i += 1

    mask = mask[:, :, 1]  # convert to greyscale
    return image, mask

def create_roi_mask(image_shape: Tuple[int, int], coordinates: List[Tuple[int, int]]) -> np.ndarray:
    """
    Create a region of interest (ROI) mask based on the given image shape and coordinates.

    Parameters:
    image_shape (tuple): The shape of the image (rows, columns).
    coordinates (list): The coordinates of the ROI polygon vertices.

    Returns:
    numpy.ndarray: The ROI mask array.

    """
    print(image_shape)
    mask_array = np.zeros(image_shape, dtype=np.uint8)
    rows, cols = mask_array.shape[:2]

    # Create a mask polygon from the coordinates
    polygon = [(x, y) for x, y in coordinates]

    # Generate a binary mask using the polygon
    rr, cc = draw.polygon(np.array([p[1] for p in polygon]), np.array(
        [p[0] for p in polygon]), (rows, cols))
    mask_array[rr, cc] = 1

    return mask_array


def poly_to_mask(labels: dict, path_data: str):
    """Convert a VIA polygon label to a mask label

    Args:
        labels (dict): Labels
        path_data (srt): Path to the image data

    Returns:
        image, mask: arrays with the image and mask
    """

    # setup variables to hold coords
    polygon_coordinates = {}  # class: coords
    polygons = []  # coords
    objects_list = []  # classes

    filename = labels["filename"]
    region_data = labels["regions"]

    # Convert each object to a polygon
    for region in region_data:
        objects, coordinates = get_region_properties(region)
        if coordinates is not None:
            polygons.append(coordinates)
            polygon_coordinates[objects] = coordinates
            objects_list.append(objects)

    # Masking the images
    image, mask = mask_image(filename, polygons, objects_list, path_data)

    return image, mask

def save_mask(filename: str, mask: np.ndarray) -> None:
    """
    Save the mask image to a file.

    Parameters:
    filename (str): The path to save the image.
    mask (numpy.ndarray): The mask image array.

    Returns:
    None
    """
    cv2.imwrite(filename, mask)


def read_mask(filename: str) -> np.ndarray:
    """
    Read the mask image from a file.

    Parameters:
    filename (str): The path to the image file.

    Returns:
    numpy.ndarray: The mask image array.
    """
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)


def read_image(filename: str) -> np.ndarray:
    """
    Read the image from a file.

    Parameters:
    filename (str): The path to the image file.

    Returns:
    numpy.ndarray: The image array.
    """
    return cv2.imread(filename)


def read_json(json_path: str,) -> dict:
    """Read a json file and return the corresponding dict

    Args:
        json_path (str): Path to the json file to read

    Returns:
        dict: Contents of the json
    """
    with open(json_path, encoding='utf-8') as f:
        d_data = json.load(f)
    return d_data


def list_to_dict(ls: list) -> dict:
    """Convert a list to a dict, automatic keygen as natural numbers"""
    return {int(i): val for i, val in enumerate(ls)}


def dict_to_list(d: dict) -> list:
    """Convert a dict to a list (drop keys)"""
    return list(d.values())


def dict_to_json(d: dict,
                 filename: str,
                 ) -> None:
    """Save a python dict to a json file

    Args:
        d (dict): The dict to save
        filename (str): Path/name of file to save (must end in .json)
    """
    if os.path.splitext(filename)[1] != '.json':
        warnings.warn('Filename must end in .json')
        return None

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=4)
    return None


def list_files(path: str,
               ext: str = '.jpg',
               include_path: bool = True,
               ) -> list:
    """Get a list of all files with certain extension in a directory

    Args:
        path (str): The path to the directory
        ext (str, optional): The chosen file extension. Defaults to '.jpg'.
        include_path (bool, optional): Whether to include the path in the list. Defaults to True.
    Returns:
        list: List of files with full paths
    """
    if include_path:
        ls_files = [os.path.join(path, i) for i in os.listdir(
            path) if os.path.splitext(i)[1] == ext]
    else:
        ls_files = [i for i in os.listdir(
            path) if os.path.splitext(i)[1] == ext]
    return ls_files
