from CVImage import CVImage
from Matcher import Matcher
import cv2
import numpy as np

match = Matcher()
img = CVImage('/home/cesar/Documentos/Computer_Vision/01/image_0')
img.read_image()
img.copy_image()
img.acquire()
h = img.new_image.shape[0]
w = img.new_image.shape[1]
n = 2  # Number of roi's
size = np.array([[w / n], [h / n]], np.int32)
start = np.array([[0], [0]], np.int32)
# Create roi
roi = img.crop_image(start, size, img.new_image)
roi_prev = img.crop_image(start, size, img.prev_image)
match.match_flann(roi, roi_prev)
print "good_matches flann", len(match.good_matches)
roiflann = roi

match.draw_matches(roiflann, match.good_matches)
cv2.namedWindow('roi', cv2.WINDOW_NORMAL)
cv2.imshow('roi', roiflann)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Draw matches


match.match(roi, roi_prev)
print "good_matches bf", len(match.good_matches)

match.draw_matches(roi, match.good_matches)
cv2.namedWindow('roi', cv2.WINDOW_NORMAL)
cv2.imshow('roi', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
