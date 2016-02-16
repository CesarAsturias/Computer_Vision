from CVImage import CVImage
from Matcher import Matcher
import cv2
import numpy as np
from VisualOdometry import VisualOdometry
import matplotlib as mplt
import matplotlib.pyplot as plt

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

match.match(roi, roi_prev)
print "good_matches bf", len(match.good_matches)

print "good_kp1", len(match.good_kp1)

print "good_kp2", len(match.good_kp2)

match.draw_matches(roi, match.good_matches)
cv2.namedWindow('roi', cv2.WINDOW_NORMAL)
cv2.imshow('roi', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Compute F
vo = VisualOdometry()
[match.good_kp1, match.good_kp2] = vo.EstimateF_multiprocessing(match.good_kp1,
                                                                match.good_kp2)
print "Estimated F",  vo.F
# Obtener matrices de cmara
vo.P_from_F(vo.F)
vo.create_P1()
print "P1", vo.cam1.P
print "P2", vo.cam2.P
# Triangulate points

scene = vo.opt_triangulation(match.good_kp1, match.good_kp2,
                             vo.cam1.P, vo.cam2.P)
point2d = vo.cam1.project(scene)
point2d_prime = vo.cam2.project(scene)
print scene[:, :3]
print point2d[:, 0]
print len(vo.correctedkpts1[0])
c_x1 = vo.correctedkpts1[0]
print c_x1
