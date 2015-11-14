from CVImage import CVImage
import pdb
from Matcher import Matcher
import cv2
from VisualOdometry import VisualOdometry

match = Matcher()
odom = VisualOdometry()
img = CVImage('/home/cesar/Documentos/Computer_Vision/01/image_0')
img.read_image()
img.copy_image()
img.acquire()

#img1 = cv2.imread('000000.png')
#img2 = cv2.imread('000001.png')
print type(img.prev_image)
match.match_flann(img.new_image, img.prev_image)
match.draw_matches(img.prev_image, match.good_matches)
cv2.imshow('image', img.prev_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
odom.FindFundamentalRansac(match.good_kp2, match.good_kp1)
print odom.F
