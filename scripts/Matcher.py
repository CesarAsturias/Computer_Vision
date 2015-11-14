import numpy as np
import cv2
import cv2.cv as cv
import time
from cv2 import KeyPoint
from cv2 import FlannBasedMatcher 

# @file Matcher.py
# @author Cesar
# @version 1.0
# Class Matcher. Implements several methods to calculate
# the keypoints and descriptors of the images, and correlate them


class Matcher(object):
    def __init__(self):
        # Initiate ORB detectors
        self.orb = cv2.ORB()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.Flann_index_lsh = 6
        self.index_params = dict(algorithm=self.Flann_index_lsh, table_number=6, key_size=12, multi_probe_level=1)
        self.search_params = dict(checks=50)
        self.flann_matcher = FlannBasedMatcher(self.index_params, self.search_params)
        self.kp1 = KeyPoint()
        self.kp2 = KeyPoint()
        self.desc1 = None
        self.desc2 = None
        self.matches = None
        self.ratio = 0.65
        self.matches1 = None
        self.matches2 = None
        self.good_matches = None
        self.good_matches = None
        self.good_kp1 = [] 
        self.good_kp2 = []

    def match(self, img_new, img_prev):
        # Compute the matches for the two images
        self.kp1, self.desc1 = self.orb.detectAndCompute(img_prev, None)
        self.kp2, self.desc2 = self.orb.detectAndCompute(img_new, None)
        self.matches = self.bf.match(self.desc1, self.desc2)
        self.matches = sorted(self.matches, key=lambda x: x.distance)

    def filter_distance(self, matches):
        # Clear matches for wich NearestNeighbor (NN) ratio is > than threshold
        dist = []
        sel_matches = []
        for i in range(0, len(matches)):
            for j in range(0,1):
                dist.append(matches[i][j].distance)
        
        thres_dist = (sum(dist)/len(dist)) * self.ratio

        # Keep only reasonable matches
        for i in range(0, len(matches)):
            if len(matches[i]) == 2:
                if (matches[i][0].distance / matches[i][1].distance) < thres_dist:
                
                    sel_matches.append(matches[i])

        
        return sel_matches

    def filter_asymmetric(self, matches1, matches2):
        # Keep only symmetric matches
        sel_matches = []
        # For every match in the forward direction, we remove those that aren't found in the other direction
        for match1 in matches1:
            for match2 in matches2:
                if self.kp2[match1[0].queryIdx] == self.kp2[match2[0].trainIdx] and self.kp1[match1[0].trainIdx] == self.kp1[match2[0].queryIdx]:
                    # We keep only symmetric matches and store the keypoints of this matches
                    sel_matches.append(match1)
                    self.good_kp2.append(self.kp2[match1[0].queryIdx].pt)
                    self.good_kp1.append(self.kp1[match1[0].trainIdx].pt)
                    break
        return sel_matches

    def filter_matches(self, matches1, matches2):

        matches1 = self.filter_distance(matches1)
        matches2 = self.filter_distance(matches2)

        return self.filter_asymmetric(matches1, matches2)

    def match_flann(self, img_new, img_prev):
        # Compute matches for the two images
        # First, keypoints and descriptors for both images
        self.kp1, self.desc1 = self.orb.detectAndCompute(img_new, None)
        self.kp2, self.desc2 = self.orb.detectAndCompute(img_prev,None)

        # Next, match:

        matches1 = self.flann_matcher.knnMatch(self.desc1, self.desc2, k=2)
        matches2 = self.flann_matcher.knnMatch(self.desc2, self.desc1, k=2)
        self.good_matches = self.filter_matches(matches1, matches2)
        print self.good_kp1[0]
        print self.good_kp2[0]

    def draw_matches(self, img, matches):
        # Draw matches in the last image
        # @param img: image
        # @param matches: a matcher object (opencv)
        # @param kp1: keypoints of the old frame
        # @param kp2: keypoints of the new frame
        # @return img: image with lines between correlated points
        for i in range(len(matches)):
            idtrain = matches[i][0].trainIdx
            idquery = matches[i][0].queryIdx
            
            point_train = self.kp2[idtrain].pt
            point_query = self.kp1[idquery].pt
            
            point_train = self.transform_float_int_tuple(point_train)
            point_query = self.transform_float_int_tuple(point_query)
            
            
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
