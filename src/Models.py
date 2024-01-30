"""
This module contains the implementation of the U-Net model for image segmentation.

The U-Net model consists of an encoder-decoder architecture with skip connections.
It is commonly used for various image segmentation tasks.

The module provides functions to create the U-Net model and its components.

Functions:
- conv_block(inputs, filters): Applies a convolutional block to the input tensor.
- unet(shape, num_classes, filters): Creates the U-Net model with the specified shape, 
                                        number of classes, and filters.

Constants:
- DROPOUT_RATE: The dropout rate used in the convolutional blocks.
- NUM_FILTERS: The number of filters used in each convolutional block.
- INTERPOLATION: The interpolation method used in the upsampling layers.
"""
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, UpSampling2D, Concatenate, Dropout, LeakyReLU
from tensorflow.keras.models import Model


DROPOUT_RATE = 0.2
NUM_FILTERS = [32, 64, 128, 256, 512]
INTERPOLATION = 'bilinear'


def conv_block(inputs: tf.Tensor,
               filters: int):
    """
    Convolutional block

    This function applies two convolutional layers with batch normalization, leaky ReLU activation,
    and dropout to the input tensor.

    Args:
        inputs (Tensor): Input tensor to the convolutional block.
        filters (int): Number of filters for the convolutional layers.

    Returns:
        Tensor: Output tensor after applying the convolutional block.
    """
    x = Conv2D(filters, (3, 3), kernel_initializer='he_normal',
               padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(DROPOUT_RATE)(x)

    x = Conv2D(filters, (3, 3), kernel_initializer='he_normal',
               padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(DROPOUT_RATE)(x)

    return x


def unet(shape: tuple[int, int, int],
         num_classes: int,
         filters: list[int] = None) -> Model:
    """
    Creates the U-Net model with the specified shape, number of classes, and filters.

    Args:
        shape (tuple[int, int, int]): Shape of the input tensor (height, width, channels).
        num_classes (int): Number of output classes.
        filters (list[int]): Filters used in each convolutional block. Defaults to NUM_FILTERS.

    Returns:
        Model: U-Net model.
    """
    if filters is None:
        filters = NUM_FILTERS

    inputs = Input(shape)

    # Encoder
    x1 = conv_block(inputs, filters[0])
    p1 = MaxPool2D((2, 2))(x1)

    x2 = conv_block(p1, filters[1])
    p2 = MaxPool2D((2, 2))(x2)

    x3 = conv_block(p2, filters[2])
    p3 = MaxPool2D((2, 2))(x3)

    x4 = conv_block(p3, filters[3])
    p4 = MaxPool2D((2, 2))(x4)

    # Bridge
    b1 = conv_block(p4, filters[4])

    # Decoder
    u1 = UpSampling2D((2, 2), interpolation=INTERPOLATION)(b1)
    c1 = Concatenate()([u1, x4])
    x5 = conv_block(c1, filters[3])

    u2 = UpSampling2D((2, 2), interpolation=INTERPOLATION)(x5)
    c2 = Concatenate()([u2, x3])
    x6 = conv_block(c2, filters[2])

    u3 = UpSampling2D((2, 2), interpolation=INTERPOLATION)(x6)
    c3 = Concatenate()([u3, x2])
    x7 = conv_block(c3, filters[1])

    u4 = UpSampling2D((2, 2), interpolation=INTERPOLATION)(x7)
    c4 = Concatenate()([u4, x1])
    x8 = conv_block(c4, filters[0])

    # Output layer
    output = Conv2D(num_classes, (1, 1), activation="softmax",
                    kernel_initializer='glorot_normal', padding='same')(x8)

    return Model(inputs, output)
