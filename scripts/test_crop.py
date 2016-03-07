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


cv2.namedWindow('Invariant', cv2.WINDOW_NORMAL)
cv2.imshow('Invariant', img.invariant_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

h = img.new_image.shape[0]
w = img.new_image.shape[1]
n = 2  # Number of roi's
size = np.array([[w / n], [h / n]], np.int32)
start = np.array([[0], [0]], np.int32)
# Create roi
roi = img.crop_image(start, size, img.new_image)
roi_prev = img.crop_image(start, size, img.prev_image)


match.match(roi, roi_prev)
print "good_matches bf", type(match.good_matches)


match.draw_matches(roi, match.good_matches)
cv2.namedWindow('roi', cv2.WINDOW_NORMAL)
cv2.imshow('roi', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Second ROI:
match.sum_coord(start[0], start[1])

print type(match.good_kp1)
print type(match.good_kp1[0])



start = np.array([[w / n], [0]])

roi = img.crop_image(start, size, img.new_image)

roi_prev = img.crop_image(start, size, img.prev_image)
# save matches
match.append_matches()
match.append_keypoints1()
match.append_keypoints2()


match.match(roi, roi_prev)


match.draw_matches(roi, match.good_matches)
cv2.namedWindow('roi', cv2.WINDOW_NORMAL)
cv2.imshow('roi', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Third  ROI:
match.sum_coord(start[0], start[1])

start = np.array([[0], [h / n]])

roi = img.crop_image(start, size, img.new_image)

roi_prev = img.crop_image(start, size, img.prev_image)
# save matches

match.append_matches()
match.append_keypoints1()
match.append_keypoints2()


match.match(roi, roi_prev)


match.draw_matches(roi, match.good_matches)
cv2.namedWindow('roi', cv2.WINDOW_NORMAL)
cv2.imshow('roi', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Fourth ROI:
match.sum_coord(start[0], start[1])

start = np.array([[w / n], [h / n]])

roi = img.crop_image(start, size, img.new_image)

roi_prev = img.crop_image(start, size, img.prev_image)
# save matches
match.append_matches()
match.append_keypoints1()
match.append_keypoints2()


match.match(roi, roi_prev)


match.draw_matches(roi, match.good_matches)
cv2.namedWindow('roi', cv2.WINDOW_NORMAL)
cv2.imshow('roi', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save matches
match.sum_coord(start[0], start[1])

match.append_matches()
match.append_keypoints1()
match.append_keypoints2()

print "len keypoints", len(match.global_kpts1)
match.global_kpts1 = np.float32(match.global_kpts1)
match.global_kpts2 = np.float32(match.global_kpts2)
print len(match.global_kpts1)



vo = VisualOdometry()

for i in range(2):
    print  len(match.global_kpts1)

    image_plane = np.zeros((h, w, 3), np.uint8)

    #image_plane= img.create_img()

    [match.global_kpts1, match.global_kpts2] = vo.EstimateF_multiprocessing(match.global_kpts1, match.global_kpts2)

    print vo.F

    print len(match.global_kpts1)

    l1 = len(match.global_kpts1)

    image_plane = match.draw_matches_np(image_plane, match.global_kpts1, match.global_kpts2)
    image_plane = match.draw_outliers_np(image_plane, vo.outlier_points_new, vo.outlier_points_prev)
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Image', image_plane)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #print "puntos deshechados", vo.outlier_points_new[:10]

img.invariant_image = match.draw_matches_np(img.invariant_image, match.global_kpts1, match.global_kpts2)
img.invariant_image = match.draw_outliers_np(img.invariant_image, vo.outlier_points_new, vo.outlier_points_prev)

plt.imshow(img.invariant_image, cmap='gray', interpolation = 'bicubic')
plt.show()
match.global_kpts1 = np.reshape(match.global_kpts1, (len(match.global_kpts1), 2))
match.global_kpts2 = np.reshape(match.global_kpts2, (len(match.global_kpts2), 2))



