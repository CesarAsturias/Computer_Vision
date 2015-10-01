import sys, os
import numpy as np
import cv2 
import cv2.cv as cv
import time

# @file CVImage.py
# @author César
# @version 1.0
# Class CVImage. It read and store the image, and implements
# several methods.

class CVImage(object):
    def __init__(self, path):
        # Initialize a number of global variables
        self.frame = None
        self.frame_size = None
        self.frame_width = None
        self.frame_height = None
        self.display_image = None
        self.grey = None
        self.prev_grey = None
        self.keystroke = None
        self.resize_window_width = 0
        self.resize_window_height = 0
        self.path = str(path)
        self.counter = 0

        # Check the number of images in the path
        self.number_images = self.count_images(self.path)
        
        # Create the main display window
        self.cv_window_name = CV_WINDOW
        cv.NamedWindow(self.cv_window_name, cv.CV_WINDOW_NORMAL)
        if self.resize_window_height > 0 and self.resize_window_width > 0:
            cv.ResizeWindow(self.cv_window_name, self.resize_window_width, self.resize_window_height)

        # Set a call back on mouse clicks on the image window
        cv.SetMouseCallback(self.cv_window_name, self.on_mouse_click, None)


    def read_image(self, path):
        # This function reads the new image from the path specifie
        # @param path : abs path to the image
        # @return img : new image



    def count_images(self, path):
        # Count the number of images in the specified path
        # @param path: string containing the path of the images
        # @return count: Number of images

        count = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
        return count

    def on_mouse_click(self, event, x, y, flags, param):
        # This function allows the user to selct a ROI using the mouse
