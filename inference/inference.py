#!/usr/bin/env python
__author__ = "David Serret"
__license__ = "GPLv3"
__version__ = "1.1"
__email__ = "david.serret.mayer<at>gmail.com"

import time
import argparse

# Use pip
import cv2
import keras  # also needs tensorflow
import numpy as np

# Custom libraries
import threaded_grabber as th

# Parsing
parser = argparse.ArgumentParser()
parser.add_argument('-t', action='store_true', help='Print timer')

parser.add_argument('-p', type=str, default="/dev/video0", help='Video port')
parser.add_argument('-w', type=str,
                    default="20180821weights.h5", help='Path to weight file')
args = parser.parse_args()

# Import the model
model = keras.models.load_model(args.w)


# The input shape is defined in traning
img_shape = [128, 96, 3]
labelnames = ["rock", "paper", "scissor"]

#  Initialize the grabber thread
th = th.ThreadedGrabber(img_shape=img_shape, port=args.p)

timer = time.time()

while(True):
    # If timer flag is enabled
    if args.t:
        print(time.time()-timer)
        timer = time.time()

    # Get image from ThreadedGrabber
    prediction, original = th.get()

    # This is the actual inference
    pred = model.predict(prediction, batch_size=None, verbose=0, steps=None)

    # Draw rock paper scissors on the original image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(original, labelnames[np.argmax(pred)], (50, 100), font, 3,
                (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(original, (
                "r: "+str(int(round(pred[0][0]*100))).zfill(3) +
                " p: "+str(int(round(pred[0][1]*100))).zfill(3) +
                " s: "+str(int(round(pred[0][2]*100))).zfill(3)),
                (50, 450), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', original)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

th.stop()
