from CVImage import CVImage
import pdb
from Matcher import Matcher
import cv2
from VisualOdometry import VisualOdometry
import numpy as np

match = Matcher()
print(match.orb.getScaleFactor())
odom = VisualOdometry()
img = CVImage('/home/cesar/Documentos/Computer_Vision/01/image_0')
img.read_image()
img.copy_image()
img.acquire()
print img.new_image.shape
n = 2
height = img.new_image.shape[0]
width = img.new_image.shape[1]
size = np.array([[width / n], [height / n]], np.int32)
start = np.array([[0], [0]], np.int32)
print size
print width
print height
for i in range(0, n):
    for j in range(0, n):
        start_temp = np.array([[start[0, 0] + i * size[0, 0]], [start[1, 0] + j * size[1, 0]]], np.int32)
        roi = img.crop_image(start_temp, size, img.new_image)
        roi_prev = img.crop_image(start_temp, size, img.prev_image)
        print start
        print start_temp
        cv2.imshow('roi', roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('roi_prev', roi_prev)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        match.match_flann(roi, roi_prev)
        print len(match.good_matches)

        match.draw_matches(roi_prev, match.good_matches)
        cv2.imshow('image', roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print i
        print j

match.match_flann(img.prev_image, img.new_image)
#print match.good_matches


#match.draw_matches(img.prev_image, match.good_matches)
#cv2.imshow('image', img.prev_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#odom.FindFundamentalRansac(match.good_kp2, match.good_kp1)
#print odom.F
