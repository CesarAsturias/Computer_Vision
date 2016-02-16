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
print vo.F
# Obtener matrices de cmara
vo.P_from_F(vo.F)
vo.create_P1()
print "P1", vo.cam1.P
print "P2", vo.cam2.P
# Triangulate points
vo.optimal_triangulation(match.good_kp2, match.good_kp1)
print "origianl points1", match.good_kp1[0:5, :]
print "corrected points1", vo.correctedkpts1[:, 0:5]
print "origianl points2", match.good_kp2[0:5, :]
print "corrected points2", vo.correctedkpts2[:, 0:5]
#print "structure", np.shape(vo.structure)
print "structure", vo.structure[:, 1]
image = vo.cam2.project(vo.structure)
print "reprojected", image[:, 0:5]

image2 = vo.cam1.project(vo.structure)
print "reprojected", image2[:, 0:5]
print type(match.good_kp1)
# pointsho = vo.make_homog(np.transpose(match.good_kp1))
# pointsho2 = vo.make_homog(np.transpose(match.good_kp2))
print np.shape(vo.correctedkpts1[0, :, :])
pointsho = vo.make_homog(np.transpose(vo.correctedkpts1[0, :,:]))
pointsho2 = vo.make_homog(np.transpose(vo.correctedkpts2[0, :, :]))
print "pointsho", pointsho[0:3, 1]

print "pointsho", pointsho2[0:3, 1]
points3d = vo.triangulate_point(pointsho[0:3, 1], pointsho2[0:3, 1],
                                vo.cam1.P, vo.cam2.P)
print points3d
point2d = vo.cam1.project(points3d)
print point2d
point2d = vo.cam2.project(points3d)
print point2d
