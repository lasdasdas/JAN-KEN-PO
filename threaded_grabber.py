#!/usr/bin/env python

__author__ = "David Serret"
__license__ = "GPLv3"
__version__ = "1.1"
__email__ = "david.serret.mayer<at>gmail.com"

import time
import cv2
import numpy as np
import threading


class ThreadedGrabber():
    """
    Using Threading library empties the V4L buffer in
    the OpenCV VideoCapture object and returns the last
    copy when queried with

    copy_ready_for_inference, original_image = obj.get()
    """
    def __init__(self, img_shape=[64*3, 48*3, 3], port="/dev/video0"):
        self.cap = cv2.VideoCapture(port)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        # Return images as arrays of zeros
        self.img_shape = img_shape
        self.prediction = np.zeros((1, img_shape[1], img_shape[0],
                                    img_shape[2]), dtype=np.float32)
        self.frame = np.zeros((img_shape[1], img_shape[0],
                               img_shape[2]), dtype=np.float32)

        # Capture thread
        self.running = True
        self.lock = threading.Lock()
        self.th = threading.Thread(target=self.run)
        self.th.start()

        empty_count = 0
        while True:
            with self.lock:
                empty_count += 1
                if not np.all(self.frame == 0):
                    break
                time.sleep(0.001)
                if empty_count > 5000:
                    raise Exception('Connection to video camara'
                                    'on port '+port+' failed.')

    def __del__(self):
        self.stop()

    def run(self):
        while self.running:
            ret, frame = self.cap.read()

            if frame is None:
                continue

            # With a lock copy to return
            with self.lock:
                self.frame = frame

    def get(self):
        with self.lock:
            # Resize to fit size
            frame_pred = cv2.resize(self.frame, (self.img_shape[0],
                                    self.img_shape[1]),
                                    interpolation=cv2.INTER_CUBIC)
            # Average substraction
            frame_pred = frame_pred.astype(np.float32)/255.0
            frame_pred -= frame_pred.mean()

            # Keras inference format
            self.prediction[0, :, :, :] = frame_pred
            return self.prediction, self.frame

    def stop(self):
        self.running = False
        if self.th.isAlive():
            self.th.join()
