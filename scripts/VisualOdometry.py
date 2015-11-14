import numpy as np
import cv2
import cv2.cv as cv
import time

# @file VisualOdometry.py
# @author Cesar
# @version 1.0
# Class VisualOdometry. This class contain the measured
# odometry. Implements the calculus of the odometry
import CVImage

class VisualOdometry(object):
    def __init__(self):
        self.F = None
        self.inlier_points_new = None
        self.inlier_points_prev = None
        self.mask = None


    def FindFundamentalRansac(self,kpts1, kpts2):
        # Compute Fundamental matrix from a set of corresponding keypoints, within a RANSAC scheme
        kpts1 = np.float32(kpts1)
        kpts2 = np.float32(kpts2)

        self.F, self.mask = cv2.findFundamentalMat(kpts1, kpts2, cv2.FM_RANSAC)

        


