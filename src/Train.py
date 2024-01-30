"""
Train the model using the provided architecture, configuration, and data.

Args:
    model_arch (function): Function that returns the model architecture.
    d_cfg (dict): Configuration dictionary.
    ls_metrics (list): List of metrics to evaluate the model.
    output_dir (str): Output directory for saving files (default: 'outputs').
    v (int): Verbosity level (default: 1).

Returns:
    tuple: Tuple containing the model history and evaluation metrics.
"""
# Generic imports
import warnings
import os
import json
from typing import Callable, Dict, List, Tuple

# Package imports
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard

# Custom imports
from src import Metrics as metrics
from src import LossFunctions as loss
from src.Evaluate import metric_performance, time_performance
from src.DataLoader import DataLoader



def train_model(model_arch: Callable,
                d_cfg: Dict,
                ls_metrics: List,
                output_dir: str = 'outputs',
                v: int = 1) -> Tuple:
    """
    Train the model using the provided architecture, configuration, and data.

    Args:
        model_arch (function): Function that returns the model architecture.
        d_cfg (dict): Configuration dictionary.
        ls_metrics (list): List of metrics to evaluate the model.
        output_dir (str): Output directory for saving files (default: 'outputs').
        v (int): Verbosity level (default: 1).

    Returns:
        tuple: Tuple containing the model history and evaluation metrics.
    """

    # Set up directories for config files, model weights and test metrics
    path_cfg_files = os.path.join(output_dir, 'cfg')
    path_model_weights = os.path.join(output_dir, 'models')
    path_scores = os.path.join(output_dir, 'metrics')
    os.makedirs(path_cfg_files, exist_ok=True)
    os.makedirs(path_model_weights, exist_ok=True)
    os.makedirs(path_scores, exist_ok=True)

    # Model config
    name = d_cfg['Name']
    # save config file
    save_json(os.path.join(path_cfg_files, f'{name}_cfg.json'), d_cfg)

    shape = d_cfg['Model']['Input_shape']
    n_classes = d_cfg['Model']['Num_classes']
    classes = d_cfg['Model']['Classes']
    batch_size = d_cfg['Model']['Batch_size']
    epochs = d_cfg['Model']['Epochs']
    lr = d_cfg['Model']['Learning_rate']
    str_loss = d_cfg['Model']['Loss_function']
    loss_fn = loss.loss_functions_map(str_loss)
    clip_val = d_cfg['Model']['Clip_value']

    d_transforms = d_cfg["Transforms"]
    d_augs = d_cfg["Augmentations"]

    v = d_cfg['Verbosity']

    if v == 1:
        print('Augmentations:', d_augs)
    if v > 1:
        print('Augmentations:', d_augs)
        print('Transforms:', d_transforms)

    image_paths = d_cfg['Data']['Train_images']
    mask_paths = d_cfg['Data']['Train_masks']

    test_images = d_cfg['Data']['Test_images']
    test_masks = d_cfg['Data']['Test_masks']

    train_ims, valid_ims, train_masks, valid_masks = train_test_split(
        image_paths, mask_paths, test_size=d_cfg['Data']['Validation_split'])

    if v >= 1:
        print('Training set size:', len(train_ims),
              'Training masks size:', len(train_masks),
              'Validation set size:', len(valid_ims),
              'Validation masks size:', len(valid_masks),
              'Test set size:', len(test_images),
              'Test masks size:', len(test_masks))

    dl_train = DataLoader(image_paths=train_ims,
                          mask_paths=train_masks,
                          n_classes=n_classes,
                          channels=(1, 1),
                          augment=True,
                          shuffle=True,
                          d_transforms=d_transforms,
                          d_augs=d_augs,
                          batch_size=batch_size,
                          seed=47).build_dataloader()

    dl_valid = DataLoader(image_paths=valid_ims,
                          mask_paths=valid_masks,
                          n_classes=n_classes,
                          channels=(1, 1),
                          augment=False,  # No augs
                          shuffle=True,
                          d_transforms=d_transforms,
                          d_augs=None,
                          batch_size=batch_size,
                          seed=47).build_dataloader()

    dl_test = DataLoader(image_paths=test_images,
                         mask_paths=test_masks,
                         n_classes=n_classes,
                         channels=(1, 1),
                         augment=False,  # No augs
                         shuffle=False,
                         d_transforms=d_transforms,
                         batch_size=batch_size,
                         seed=47).build_dataloader()

    # Build Model
    model = model_arch(shape, n_classes)
    model.compile(loss=loss_fn, optimizer=tf.keras.optimizers.Adam(
        lr, clipvalue=clip_val), metrics=ls_metrics)

    timetaken = metrics.timecallback()  # training times per epoch
    callbacks = [
        ModelCheckpoint(os.path.join(path_model_weights, "{}_weights.h5".format(name)),
                        monitor='val_loss', verbose=0, save_best_model=True),
        ReduceLROnPlateau(monitor="val_loss", patience=10,
                          factor=0.1, verbose=v, min_lr=1e-6),
        EarlyStopping(monitor="val_loss", patience=15, verbose=1),
        TensorBoard(log_dir='logs'),
        timetaken
    ]

    # Train
    model_history = model.fit(dl_train,
                              validation_data=dl_valid,
                              epochs=epochs,
                              callbacks=callbacks,
                              workers=4,
                              use_multiprocessing=True,
                              verbose=v,
                              )

    # Evaluate
    model.load_weights(os.path.join(
        path_model_weights, f'{name}_weights.h5'))
    test_masks = model.predict(
        dl_test, verbose=1, workers=4, use_multiprocessing=True)
    test_masks = tf.convert_to_tensor(test_masks)

    for _, gt_masks in dl_test:
        break

    # Model performance
    d_output = {
        'Times': time_performance(timetaken.times),
        'Dice': metric_performance(gt_masks, test_masks, metric='dice', classes=classes),
        'IoU': metric_performance(gt_masks, test_masks, metric='iou', classes=classes),
        'Tversky': metric_performance(gt_masks, test_masks, metric='tversky', classes=classes),
    }

    # need to convert from np.float32 to float for serialisation
    d_history = {key: [float(i) for i in values]
                 for key, values in model_history.history.items()}
    d_output['Training_data'] = d_history

    save_json(os.path.join(path_scores, f'{name}_scores.json'),
              d_output)  # save scores to file

    return model_history, d_output


def save_json(filename: str, dict_save: dict):
    """
    Save a dictionary to a json file

    Args:
        filename (str): filename of json
        dict_save (dict): dictionary to save

    Returns:
        int: Return 0 if successful, else return 1
    """
    if os.path.splitext(filename)[1] != '.json':
        warnings.warn('save_config: Filename must end in .json', UserWarning)
        return 1

    with open(filename, 'w', encoding='utf-8') as output:
        json.dump(dict_save, output, indent=4)
        return 0
