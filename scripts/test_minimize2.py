from CVImage import CVImage
from Matcher import Matcher
import cv2
import numpy as np
from VisualOdometry import VisualOdometry
import matplotlib as mplt
import matplotlib.pyplot as plt
import time

match = Matcher()
img = CVImage('/home/cesar/Documentos/Computer_Vision/01/image_0')
img.read_image()
img.copy_image()
img.acquire()
h = img.new_image.shape[0]
print "IMAGE HEIGHT", h
w = img.new_image.shape[1]
print "IMAGE WITH", w
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
print "shape point estimated", np.shape(point2d)
print np.shape(match.good_kp1)
point2d_nh = np.delete(point2d, 2, 0)
print point2d_nh[:, :10]
print np.shape(match.good_kp1[:, :10])
print match.good_kp1[:10, :].transpose()
# Working on compute the distance: the following line don't work
print np.subtract(np.transpose(point2d_nh[:, :5]), match.good_kp1[:5, :])

# With the residual function

# Pass the estimated points, point2d, and the measured ones, match.good_kp1
error = vo.residual(match.good_kp1, point2d)
print np.shape(error)
print np.shape(error.ravel())
print "CAMARA P2", vo.cam2.P
# Prior error
vec = None
vec = np.hstack(vo.cam2.P)
vec2 = np.delete(vo.structure, 3, 0)
vec2 = vec2.reshape(-1)
vec = np.append(vec, vec2)
prior_error = vo.functiontominimize(vec, match.good_kp1, match.good_kp2)
# Test minimize function
t0 = time.time()
param_opt, param_cov = vo.optimize_F(match.good_kp1, match.good_kp2)
t1 = time.time()
print "Optimization took {} seconds".format(int(t1 - t0))
print vo.cam2.P
print param_opt[:9].reshape((3, 3))

# Plot the residuals
plt.plot(prior_error, 'k')
posterior_error = vo.functiontominimize(param_opt, match.good_kp1, match.good_kp2)
plt.plot(posterior_error, 'r')
plt.show()
