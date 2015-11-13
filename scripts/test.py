from CVImage import CVImage
import pdb
from Matcher import Matcher
import cv2

match = Matcher()
img1 = cv2.imread('000000.png')
img2 = cv2.imread('000001.png')

match.match_flann(img1, img2)
match.draw_matches(img1, match.good_matches)
cv2.imshow('image', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
