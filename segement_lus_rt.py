"""
This script performs real-time lung ultrasound segmentation using a pre-trained model.
It captures video input from a webcam or an external video device, applies the segmentation model,
and overlays the segmentation mask on the video frames in real-time.

The script also provides options to save the processed video and adjust the overlay transparency.

Note: Make sure to install the required dependencies (OpenCV, TensorFlow, Matplotlib, and Pandas) before running the script.
"""
from time import perf_counter

import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd

from src import Models as models
from src.DataLoader import DataLoader
from src import Visualise as vis

# MODEL PARAMETERS
CLASS_NAMES = ['Ribs', 'Pleural line', 'A-line', 'B-line', 'B-line confluence']
CLASSES = {i+1: CLASS_NAMES[i]
           for i in range(len(CLASS_NAMES))}  # map indices to classes

MODEL_FILE = 'model_lus.h5'

MODEL_INPUT_SHAPE = (256, 256, 1)
NUM_CLASSES = len(CLASS_NAMES) + 1
TRANSFORMS = {
    'crop': (486, 120, 1605, 915),  # (x1, y1, x2, y2)
    'resize': (256, 256),  # (width, height)
    'one-hot': True,
}

""" VIDEO INPUT PARAMETERS"""
VIDEO_INPUT_SOURCE = 0  # 0 is usually a webcam and 1/2 is an external video device. Can also set to be a video file
VIDEO_INPUT_RESOLUTION = (1920, 1080)
VIDEO_INPUT_FPS = 30


""" VISUALISATION PARAMETERS """
OVERLAY = True
OVERLAY_TRASNPARENCY = 0.5
CMAP = plt.cm.tab10.colors

""" OUTPUT PARAMETERS """""
OUTPUT_FILE = None  # 'test_i.avi'
OUTPUT_FPS = 15

###########################
# Define the model
model = models.unet(MODEL_INPUT_SHAPE, NUM_CLASSES,
                    filters=[32, 64, 128, 256, 512])
model.load_weights(MODEL_FILE)


# Define dataloader (for rescaling)
dl = DataLoader(d_transforms=TRANSFORMS)


def prep_frame(frame, resize=True, mask_im=None):
    # pre-process frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if mask_im:
        frame = cv2.bitwise_and(frame, frame, mask=mask_im)

    # convert (h,w) -> (1,h,w,1)
    frame = tf.convert_to_tensor(frame)[tf.newaxis, :, :, tf.newaxis]

    if resize:
        frame = dl.resize_and_rescale(frame)  # resize input frame

    return frame


def run_inference_video(model, input=0, save=None):

    print('Starting model inference... (press q to quit)')
    global OVERLAY
    global OVERLAY_TRASNPARENCY

    # Set up data structures to store timing information
    d_times = {'preprocessing': [], 'inference': [],
               'display': [], 'total': [], }
    d_params = {'show': True}

    # Set up video processing
    capture = cv2.VideoCapture(input)
    capture.set(cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_INPUT_RESOLUTION[0])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_INPUT_RESOLUTION[1])
    capture.set(cv2.CAP_PROP_FPS, VIDEO_INPUT_FPS)

    success, frame0 = capture.read()  # Read the first frame
    if not success:
        print('Unable to read video at input {}'.format(input))

    # Original size of video feed frame
    size = frame0.shape

    # Shape of cropped region
    reshape = (TRANSFORMS['crop'][2] - TRANSFORMS['crop'][0],
               TRANSFORMS['crop'][3] - TRANSFORMS['crop'][1])  # (x,y)

    # Padding to add to cropped region to get back to original size
    pad = (TRANSFORMS['crop'][1],
           size[0] - TRANSFORMS['crop'][3],
           TRANSFORMS['crop'][0],
           size[1] - TRANSFORMS['crop'][2])  # (top, bottom, left, right)

    print(size, reshape, pad)

    if save:
        out = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*'XVID'),
                              OUTPUT_FPS, (size[1], size[0]))

    while success:  # frame read successfully

        t0 = perf_counter()

        # pre-processing
        # crop to roi and resize to input shape of model
        frame = prep_frame(frame0)

        t1 = perf_counter()

        # run inference
        pred_mask = model.predict(frame, verbose=0)

        t2 = perf_counter()

        if d_params['show']:

            pred_mask = pred_mask[0, :, :, :]
            # convert from 1-hot encoded to 2d array
            pred_mask = pred_mask.argmax(axis=2)
            pred_mask = vis.seg_to_rgb(pred_mask, cm=CMAP)

            # Upscale segmentation prediction to original size to overlay onto the frame
            pred_mask = cv2.resize(pred_mask.astype(
                'float32'), reshape, interpolation=cv2.INTER_AREA)
            pred_mask = cv2.copyMakeBorder(
                pred_mask, pad[0], pad[1], pad[2], pad[3], cv2.BORDER_CONSTANT)

            # Add the overlay onto the original frame
            frame = frame0/255
            if OVERLAY:
                pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_RGB2BGR)
                # overlay segmentations to image
                frame = cv2.addWeighted(
                    frame, 1.0, pred_mask, OVERLAY_TRASNPARENCY, gamma=0, dtype=cv2.CV_32F)

            cv2.imshow('Model output', frame)

            t3 = perf_counter()

            if save:
                frame = cv2.normalize(
                    frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                out.write(frame)

        # quit when 'q' pressed
        key = cv2.waitKey(1)
        if key == ord('q'):  # quit when 'q' pressed
            break
        if key == ord('w'):
            if OVERLAY_TRASNPARENCY+0.1 < 1:
                OVERLAY_TRASNPARENCY += 0.1
            else:
                OVERLAY_TRASNPARENCY = 1

        elif key == ord('s'):
            if OVERLAY_TRASNPARENCY-0.1 > 0:
                OVERLAY_TRASNPARENCY -= 0.1
            else:
                OVERLAY_TRASNPARENCY = 0

        d_times['preprocessing'].append(t1-t0)
        d_times['inference'].append(t2-t1)
        d_times['display'].append(t3-t2)
        d_times['total'].append(t3-t0)

        success, frame0 = capture.read()  # read next frame

    # Close streams
    capture.release()
    if save:
        out.release()

    cv2.destroyAllWindows()

    df_scores = pd.DataFrame(d_times)
    df_scores.to_csv('timing_performance.csv')
    print(df_scores[1:].describe())


def click_event(event, x, y, flags, param):
    global OVERLAY

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        # Turn overlay on/off
        if OVERLAY is False:
            OVERLAY = True
        else:
            OVERLAY = False


if __name__ == '__main__':

    # Define the windows

    # Main window
    cv2.namedWindow("Model output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Model output", 1920, 1080)
    cv2.setMouseCallback('Model output', click_event)

    # Legend window
    cv2.namedWindow("Legend", cv2.WINDOW_NORMAL)
    legend = vis.create_cv2_legend(CLASSES.values(), cmap=[
                                   tuple(int(i*255) for i in c) for c in CMAP])
    cv2.imshow("Legend", legend)

    # Run model
    run_inference_video(model, input=VIDEO_INPUT_SOURCE, save=OUTPUT_FILE)
