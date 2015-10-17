import numpy as np
import cv2
import cv2.cv as cv
import time
from cv2 import KeyPoint

# @file Matcher.py
# @author Cesar
# @version 1.0
# Class Matcher. Implements several methods to calculate
# the keypoints and descriptors of the images, and correlate it

class Matcher(object):
    def __init__(self):
        # Initiate ORB detector
        self.orb = cv2.ORB()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.kp1 = KeyPoint()
        self.kp2 = KeyPoint()
        self.desc1 = None
        self.desc2 = None
        self.matches = None


    def match(self, img_new, img_prev):
        # Compute the matches for the two images
        self.kp1, self.desc1 = self.orb.detectAndCompute(img_prev, None)
        self.kp2, self.desc2 = self.orb.detectAndCompute(img_new, None)
        self.matches = self.bf.match(self.desc1, self.desc2)
        self.matches = sorted(self.matches, key=lambda x: x.distance)

    def draw_matches(self, img):
        # Draw matches in the last image
        # @param img: image
        # @param matches: a matcher object (opencv)
        # @param kp1: keypoints of the old frame
        # @param kp2: keypoints of the new frame
        # @return img: image with lines between correlated points
        for i in range(len(self.matches)):
            idtrain = self.matches[i].trainIdx
            idquery = self.matches[i].queryIdx
            print idtrain
            print idquery
            point_train = self.kp2[idtrain].pt
            point_query = self.kp1[idquery].pt
            print point_train
            print point_query
            point_train = self.transform_float_int_tuple(point_train)
            point_query = self.transform_float_int_tuple(point_query)
            print point_train
            print point_query
            cv2.line(img, ((point_train[0]), (point_train[1])),((point_query[0]), (point_query[1])), (255, 0, 0))
            
        return img

    def transform_float_int_tuple(self, input_tuple):
        output_tuple = [0, 0]
        if not input_tuple is None:
            for i in range(0, len(input_tuple)):
                output_tuple[i] = int(input_tuple[i])
        else:
            return input_tuple

        return output_tuple


