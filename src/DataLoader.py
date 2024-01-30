'''
Custom Tensorflow DataLoader object. 
Uses TensorFlow.image API to do pre-processing and 'on the fly' image augmentation
'''
from typing import List, Tuple
import warnings
import random

import tensorflow as tf
import tensorflow_addons as tfa  # rotations
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE


class DataLoader(object):
    """A TensorFlow Dataset API based loader for semantic segmentation problems.
    Forked from: https://github.com/HasnainRaz/SemSegPipeline#semseg-pipeline
    """

    def __init__(self,
                 image_paths: List[str] = None,
                 mask_paths: List[str] = None,
                 n_classes: int = 2,
                 channels: Tuple[int, int] = (3, 3),
                 batch_size: int = 16,
                 augment: bool = True,
                 shuffle: bool = True,
                 d_transforms: dict = None,
                 d_augs: dict = None,
                 seed: int = 42,
                 ):
        """Initialize a TensorFlow Dataset loader for semantic segmentation

        Args:
            image_paths (List[str]): List of paths to images.
            mask_paths (List[str]): List of paths to masks (segmentation masks)
            n_classes (int): The number of classes to segment 
            channels (Tuple[int, int], optional): 
                    Tuple of ints, first element is number of channels in images,
                    second is the number of channels in the mask image (needed to
                    correctly read the images into tensorflow and apply augmentations).
                    Defaults to (3, 3).
            batch_size (int, optional): The batch size to use. Defaults to 16.
            augment (bool, optional): Whether to apply agumentations to the DS. Defaults to True.
            shuffle (bool, optional): Whether to shuffle the DS. Defaults to True.
            d_transforms (dict, optional): 
                Transforamtions to apply (resize, crop). Defaults to None.
            d_augs (dict, optional): 
                Augmentations to apply (hflip, rotation, brightness, contrast). Defaults to None.
            seed (int, optional): Used as the seed for RNG in the data pipeline. Defaults to None.
        """

        self.resize_interpolation = 'bilinear'

        # Args
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.class_values = np.arange(n_classes)

        # Optional args w/ defauls
        self.channels = channels
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle

        # Opt args w/o defaults
        if seed is None:
            self.seed = random.randint(0, 1000)
        else:
            self.seed = seed

        # Setup dicts for pre-processing trasnformations and augmentations
        self.d_transforms = {'crop': None,  # (x1, y1, x2, y2)
                             'resize': (256, 265),  # (width, height)
                             'one-hot': False,
                             'roi_mask': None,
                             }

        self.d_augs = {'hflip': True,  # bool
                       'rotate': None,  # degrees: float
                       'brightness': None,  # delta: float
                       'contrast': None,  # (min: float, max: float)
                       # (crop_fraction: float, resize_fraction: float)
                       'depth': None,
                       'tgc': None,  # bool
                       }

        # Add in user settings
        if d_transforms is not None:
            for key in d_transforms.keys():  # check keys
                if key not in list(self.d_transforms.keys()):
                    warnings.warn(f"'{key}' not a valid transform. \
                        Choose from {list(self.d_transforms.keys())}")
            if 'one-hot' in d_transforms.keys():
                self.d_transforms['one-hot'] = d_transforms['one-hot']
            if 'crop' in d_transforms.keys():
                self.d_transforms['crop'] = d_transforms['crop']
            if 'resize' in d_transforms.keys():
                self.d_transforms['resize'] = d_transforms['resize']
            if 'roi_mask' in d_transforms.keys():
                self.d_transforms['roi_mask'] = d_transforms['roi_mask']

        if d_augs is not None:
            if self.augment == None:
                warnings.warn(
                    "d_augs provided but augment = False. Ignoring augmentations")
            for key in d_augs.keys():  # check keys
                if key not in list(self.d_augs.keys()):
                    warnings.warn("'{}' not a valid transform. Choose from {}"
                                  .format(key, list(self.d_augs.keys())),
                                  )

            if 'hflip' in d_augs.keys():
                self.d_augs['hflip'] = d_augs['hflip']
            if 'rotate' in d_augs.keys():
                self.d_augs['rotate'] = d_augs['rotate']
            if 'brightness' in d_augs.keys():
                self.d_augs['brightness'] = d_augs['brightness']
            if 'contrast' in d_augs.keys():
                self.d_augs['contrast'] = d_augs['contrast']
            if 'depth' in d_augs.keys():
                self.d_augs['depth'] = d_augs['depth']
            if 'tgc' in d_augs.keys():
                self.d_augs['tgc'] = d_augs['tgc']

        # Create a random number generator.
        self.rng = tf.random.Generator.from_seed(seed, alg='philox')

        # Set up other class variables
        self._define_vars()

    def _define_vars(self):
        """
        Set up class variables
        """

        self.im_size = self.d_transforms['resize']

        # resize roi mas
        if self.d_transforms['roi_mask'] is not None:

            # convert to tensor
            self.d_transforms['roi_mask'] = tf.convert_to_tensor(
                self.d_transforms['roi_mask'], dtype=tf.float32)
            # add batch dim
            self.d_transforms['roi_mask'] = tf.image.resize(
                self.d_transforms['roi_mask'], self.im_size)

            print('roi mask', self.d_transforms['roi_mask'].shape)

        # TGC augmentations vars
        # value for sigma of Gaussian from which darkness of TGC stripes are generated
        self.tgc_darkness_sigma = 0.1
        self.tgc_n_lines = 8  # number of lines to use in TGC mask
        _, self.tgc_meshgrid_Y = np.meshgrid(
            np.arange(0, self.im_size[0]), np.arange(0, self.im_size[0]))

    def _load_data(self, image_paths, mask_paths):
        """
        Reads image and mask files 
        """
        image_content = tf.io.read_file(image_paths)
        mask_content = tf.io.read_file(mask_paths)

        images = tf.image.decode_png(image_content, channels=self.channels[0])
        masks = tf.image.decode_png(mask_content, channels=self.channels[1])

        return images, masks

    def _one_hot_encode(self, mask):
        """
        Converts mask to a one-hot encoding specified by the semantic map.
        """
        one_hot_map = []
        for class_ in self.class_values:
            class_map = tf.reduce_all(tf.equal(mask, class_), axis=-1)
            one_hot_map.append(class_map)
        one_hot_map = tf.stack(one_hot_map, axis=-1)

        return one_hot_map

    def _gaussian(self, x: float, mu: float, sig: float) -> float:
        """Gaussian distribution function"""
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def resize_and_rescale(self, image, mask=None):
        """ Resize and scale image"""

        # Crop
        if self.d_transforms['crop'] is not None:
            (x1, y1, x2, y2) = self.d_transforms['crop']
            image = tf.image.crop_to_bounding_box(image, y1, x1, y2-y1, x2-x1)

            if mask is not None:
                mask = tf.image.crop_to_bounding_box(
                    mask, y1, x1, y2-y1, x2-x1)

        # Resize
        if self.d_transforms['resize'] is not None:
            (h, w) = self.d_transforms['resize']
            image = tf.image.resize(
                image, [h, w], method=self.resize_interpolation)

            if mask is not None:
                # use nearest neighbour interpolation for mask
                mask = tf.image.resize(mask, [h, w], method='nearest')

        # Apply roi mask
        if self.d_transforms['roi_mask'] is not None:
            image = tf.multiply(image, self.d_transforms['roi_mask'])

        # Convert dtypes
        image = tf.cast(image, tf.float32) / 255.0

        if mask is not None:
            mask = tf.cast(mask, tf.uint8)
            return image, mask
        else:
            return image

    def augment_images(self, image, mask, seed):

        # Make a new seed
        seed = self.rng.uniform_full_int([2], dtype=tf.int32)

        # IMAGE AND MASK TRANSFORMS
        # Randomly flip the image and mask
        if self.d_augs['hflip'] == True:
            image = tf.image.stateless_random_flip_left_right(
                image, seed=seed)
            mask = tf.image.stateless_random_flip_left_right(
                mask, seed=seed)

        # Random rotation
        if self.d_augs['rotate'] is not None:
            theta0 = self.d_augs['rotate']

            rot_degrees = tf.random.normal(
                [], stddev=0.5)*theta0  # genarate a random angle

            image = tfa.image.rotate(
                image, rot_degrees*np.pi/180)  # rotation in radians
            mask = tfa.image.rotate(mask, rot_degrees*np.pi/180)

        # Random crop and resize
        if self.d_augs['depth'] is not None:

            # params
            # minumum crop width relative to size of image i.e. 0.2 -> crop to 20%
            crop_minsize = self.d_augs['depth'][0]
            # minimum scale for which to zoom and pad images i.e. 0.2 -> zoom to 20% origianl size and pad to fill
            zoom_minscale = self.d_augs['depth'][1]

            # genarate random value for cropping size and scaling
            # size for random crop (minsize to 100%)
            random_size = int(
                image.shape[1]*self.rng.uniform([], minval=crop_minsize, maxval=1, dtype=tf.float32))
            random_scale = self.rng.uniform(
                [], minval=zoom_minscale, maxval=1, dtype=tf.float32)

            # crop image to random window
            image = tf.image.stateless_random_crop(
                image, size=(random_size, random_size, 1), seed=seed)
            mask = tf.image.stateless_random_crop(
                mask, size=(random_size, random_size, 1), seed=seed)

            # randomly resize cropped area and then pad to correct dimensions
            scale = int(self.d_transforms['resize'][0] * random_scale)
            pad = self.d_transforms['resize'][0] - scale

            image = tf.image.resize(image, [scale, scale])
            mask = tf.image.resize(mask, [scale, scale])

            image = tf.pad(image,
                           paddings=[[0, pad],  # pad below in y
                                     [int(pad)//2, int(pad)//2+int(pad) %
                                      2],  # pad so image cental in x
                                     [0, 0]  # channels
                                     ],
                           mode="CONSTANT", constant_values=0)

            mask = tf.pad(mask,
                          paddings=[[0, pad],  # pad below in y
                                    [int(pad)//2, int(pad)//2+int(pad) %
                                     2],  # pad so image cental in x
                                    [0, 0]  # channels
                                    ],
                          mode="CONSTANT", constant_values=0)

        # Time gain compensation (TGC) augmentation
        if self.d_augs['tgc'] == True:
            sigma = self.im_size[0] / (self.tgc_n_lines-1)

            # Initialise with zeros
            tgc_filter = np.zeros(self.im_size)

            for i in range(self.tgc_n_lines):  # add in each TGC line
                darkness = tf.abs(tf.random.normal(
                    [], stddev=self.tgc_darkness_sigma))  # random darkness
                # add each line sequentially
                tgc_filter += self._gaussian(self.tgc_meshgrid_Y,
                                             i*sigma, sigma/2) * darkness

            tgc_filter = tf.convert_to_tensor(tgc_filter, dtype=tf.float32)
            # apply filter to image
            image = image + tf.expand_dims(tgc_filter, 2)

        # IMAGE ONLY TRANSFORMS
        # Random brightness.
        if self.d_augs['brightness'] is not None:
            delta = self.d_augs['brightness']
            image = tf.image.stateless_random_brightness(
                image, max_delta=delta, seed=seed)

        # Random contrast
        if self.d_augs['contrast'] is not None:
            (lower, upper) = self.d_augs['contrast']
            image = tf.image.stateless_random_contrast(
                image, lower=lower, upper=upper, seed=seed)

        # image = tf.clip_by_value(image, 0, 1)
        return image, mask

    def process_images(self, image_label, seed):

        image, mask = image_label

        # Resize and rescale
        image, mask = self.resize_and_rescale(image, mask)

        # Augment
        if self.augment:
            image, mask = self.augment_images(image, mask, seed)

        # One-hot encode the mask
        if self.d_transforms['one-hot'] is True:
            mask = self._one_hot_encode(mask)

        mask = tf.cast(mask, tf.float32)

        # mask = tf.cast(mask, tf.uint8)
        return image, mask

    def build_dataloader(self):
        """
        Reads data, normalizes and shuffles, applies augmentations, then batches the dataset. 
        """

        # Create dataset out of the 2 files:
        data = tf.data.Dataset.from_tensor_slices(
            (self.image_paths, self.mask_paths))

        # Parse images and labels
        data = data.map(self._load_data, num_parallel_calls=AUTOTUNE)

        # Create a `Counter` object and zip it with the training set.
        counter = tf.data.experimental.Counter()
        data = tf.data.Dataset.zip((data, (counter)))

        # Shuffle and augment
        # BUFFER_SIZE = random.randint(1, len(self.image_paths))
        BUFFER_SIZE = len(self.image_paths)
        # print('BUFFER_SIZE: ', BUFFER_SIZE)

        if self.shuffle:
            data = (data
                    .cache()
                    .map(self.process_images, num_parallel_calls=AUTOTUNE)
                    .shuffle(BUFFER_SIZE)
                    .batch(self.batch_size)
                    .prefetch(buffer_size=AUTOTUNE)
                    )
        else:
            data = (data
                    .cache()
                    .map(self.process_images, num_parallel_calls=AUTOTUNE)
                    .batch(self.batch_size)
                    .prefetch(buffer_size=AUTOTUNE)
                    )

        return data
