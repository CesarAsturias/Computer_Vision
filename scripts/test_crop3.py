from CVImage import CVImage
from Matcher import Matcher
import numpy as np
from VisualOdometry import VisualOdometry
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def correlate_roi(match, img, size, start):
    # This function correlates two images using Matcher and CVImage
    # @param match: Matcher object
    # @param img: CVImage object
    # @param size: size of the region of interest
    # @param start: coordinates of the origin
    # @return match.good_kp1: keypoints founded in the current roi
    # @return match.good_kp2: keypoints founded in the previous roi

    roi = img.crop_image(start, size, img.new_image)
    roi_prev = img.crop_image(start, size, img.prev_image)
    match.match(roi, roi_prev)
    print type(match.curr_kp[0])
    print len(match.curr_kp)
    print len(match.prev_kp)

    # Translate keypoints to their actual position
    match.sum_coord(start[0], start[1])
    # Store the keypoints and the matches
    match.append_global()


def get_number_keypoints(match):
    # returns the total keypoints encountered
    return len(match.global_kpts1)


def plot_same_figure(match, img):
    # Plot the keypoints in each image on the same figure

    global_kpts1 = np.reshape(match.global_kpts1, (len(match.global_kpts1), 2))
    global_kpts2 = np.reshape(match.global_kpts2, (len(match.global_kpts2), 2))
    x1 = global_kpts1[:, 0]
    y1 = global_kpts1[:, 1]
    x2 = global_kpts2[:, 0]
    y2 = global_kpts2[:, 1]
    fig = plt.figure(figsize=(20, 20))
    a = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(img.new_image, cmap='gray', interpolation='bicubic')
    plt.plot(x1, y1, 'r*')
    a.set_title('Current Image')
    a = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(img.prev_image, cmap='gray', interpolation='bicubic')
    plt.plot(x2, y2, 'g*')
    a.set_title('Previous Image')
    plt.show()


def plot_together(match, img):
    # Store the result and plot
    plt.imshow(img.new_image, cmap='gray', interpolation='bicubic')
    global_kpts1 = np.reshape(match.global_kpts1, (len(match.global_kpts1), 2))
    global_kpts2 = np.reshape(match.global_kpts2, (len(match.global_kpts2), 2))
    x1 = global_kpts1[:, 0]
    y1 = global_kpts1[:, 1]
    plt.plot(x1, y1, 'r*')
    x2 = global_kpts2[:, 0]
    y2 = global_kpts2[:, 1]
    plt.plot(x2, y2, 'k*')
    plt.show()


def plot_together_np(array1, array2, img):
    # Store the result and plot
    plt.imshow(img.new_image, cmap='gray', interpolation='bicubic')
    global_kpts1 = np.reshape(array1, (len(array1), 2))
    global_kpts2 = np.reshape(array2, (len(array2), 2))
    x1 = global_kpts1[:, 0]
    y1 = global_kpts1[:, 1]
    plt.plot(x1, y1, 'r*')
    x2 = global_kpts2[:, 0]
    y2 = global_kpts2[:, 1]
    plt.plot(x2, y2, 'k*')
    plt.show()


def plot_one(match, img):
    # Store the result and plot
    plt.imshow(img.new_image, cmap='gray', interpolation='bicubic')
    global_kpts1 = np.reshape(match.global_kpts1, (len(match.global_kpts1), 2))
    x1 = global_kpts1[:, 0]
    y1 = global_kpts1[:, 1]
    plt.plot(x1, y1, 'r*')
    plt.show()


def plot_one_np(array1, img):
    # Store the result and plot
    plt.imshow(img.new_image, cmap='gray', interpolation='bicubic')
    array1 = np.reshape(array1, (len(array1), 2))
    x1 = array1[:, 0]
    y1 = array1[:, 1]
    plt.plot(x1, y1, 'r*')
    plt.show()


def plot_save(match, img):
    # Store the result and plot
    px = img.new_image.shape[0]
    py = img.new_image.shape[1]

    dpi = 140
    size = (py / np.float(dpi), px / np.float(dpi))

    fig = plt.figure(figsize=size, dpi=dpi)
    plt.imshow(img.new_image, cmap='gray', interpolation='bicubic')
    global_kpts1 = np.reshape(match.global_kpts1, (len(match.global_kpts1), 2))
    x1 = global_kpts1[:, 0]
    y1 = global_kpts1[:, 1]
    plt.plot(x1, y1, 'r*')
    plt.show()
    # plt.savefig('test.png', dpi=100)


def get_structure(match, img, vo):
    # Get structure of the scene
    # @param match: Matcher object
    # @param img: CVImage object
    # @param vo: VisualOdometry object
    vo.P_from_F(vo.F)
    vo.create_P1()

    # Triangulate points
    scene = vo.opt_triangulation(match.global_kpts1, match.global_kpts2,
                                 vo.cam1.P, vo.cam2.P)

    return scene


def run():
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

    # First roi
    correlate_roi(match, img, size, start)

    # Second roi
    start = np.array([[w / n], [0]])
    correlate_roi(match, img, size, start)

    # Third roi
    start = np.array([[0], [h / n]])
    correlate_roi(match, img, size, start)

    # Last roi
    start = np.array([[w / n], [h / n]])
    correlate_roi(match, img, size, start)

    # We have stored two times every original keypoint (curr_kp, prev_kp)
    match.curr_kp = match.curr_kp[::2]
    match.prev_kp = match.prev_kp[::2]

    # The same applies for the descriptors
    match.curr_dsc = match.curr_dsc[::2]
    match.prev_dsc = match.prev_dsc[::2]

    print match.curr_kp[0].pt
    print match.global_kpts1[0]
    # Print the total number of keypoints encountered
    print("Total number of keypoints encountered: {}".format(get_number_keypoints(match)))

    # Test the plot_same_figure function
    # plot_same_figure(match, img)
    # plot_one(match, img)
    # plot_save(match, img)

    # Get Fundamental Matrix
    vo = VisualOdometry()
    print "Type of match.global_kpts1: ", type(match.global_kpts1)
    match.global_kpts1, match.global_kpts2 = vo.EstimateF_multiprocessing(match.global_kpts2, match.global_kpts1)
    # plot_one(match, img)
    # plot_one_np(vo.outlier_points_new, img)
    # plot_together_np(match.global_kpts1, vo.outlier_points_new, img)
    print("Total number of keypoints encountered: {}".format(get_number_keypoints(match)))

    # Triangulate. To get the actual movement of the camera we are "swapping"
    # the scene. The first camera is cam1.P, the first keypoints are
    # global_kpts1. On the other hand, the second camera is cam2.P and the
    # second keypoints are global_kpts2
    scene = get_structure(match, img, vo)
    print "ESCENA", scene[:, :20]
    print "PROYECCION EN SEGUNDA", vo.cam1.project(scene[:, :20])
    print "SEGUNDA", match.global_kpts1[:20]
    print "CORREGIDOS SEGUNDA", vo.correctedkpts1[:, :20]
    print "PROYECCION EN PRIMERA", vo.cam2.project(scene[:, :20])
    print "PRIMERA", match.global_kpts2[:20]
    print "CORREGIDOS EN PRIMERA", vo.correctedkpts2[:, :20]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(scene[0], scene[1], scene[2], 'ko')
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    run()
