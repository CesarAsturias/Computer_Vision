import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def plot_image(img):
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.show()


def plot_two_points(points1, points2, img):
    plt.imshow(img.prev_image, cmap='gray', interpolation='bicubic')
    x1 = points1[:, 0]
    y1 = points1[:, 1]
    plt.plot(x1, y1, 'r*')
    x2 = points2[:, 0]
    y2 = points2[:, 1]
    plt.plot(x2, y2, 'k*')
    plt.show()


def plot_matches(match, img):
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


def plot_scene(scene):
    # Plot the 3D structure of the scene
    # @param scene: 3D points of the scene

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(scene[0], scene[1], scene[2], 'ko')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()


def correlate_roi(match, img, size, start):

    roi = img.crop_image(start, size, img.new_image)
    roi_prev = img.crop_image(start, size, img.prev_image)
    match.match(roi, roi_prev)
    # Check if there are any match
    if match.is_minkp and match.is_minmatches:
        match.sum_coord(start[0], start[1])
        match.append_global()


def correlate_image(match, img, nv_rois, nh_rois):
    # Correlate an image by dividing it in several windows
    # @param match: Matcher object
    # @param img: CVImage object
    # @param nv_rois: Number of rois (vertical dimension)
    # @param nh_rois: Number of rois (horizantal dimension)
    # @return p1, p2: Ground plane points (previous and current) as a numpy
    # array.

    size = np.array([[img.frame_width / nh_rois], [img.frame_height / nv_rois]],
                    np.int32)  # Size of rois

    start = np.array([[0], [0]], np.int32)
    points_plane = []

    for i in range(nv_rois):
        for j in range(nh_rois):
            start = np.array([[j * size[0]], [i * size[1]]])
            correlate_roi(match, img, size, start)
            # Save the ground plane points
            if i == 1 and j > 2 and j < 6:
                points_plane.append([match.good_kp1, match.good_kp2])

    match.curr_kp = match.curr_kp[::2]
    match.prev_kp = match.prev_kp[::2]
    match.curr_dsc = match.curr_dsc[::2]
    match.prev_dsc = match.prev_dsc[::2]

    p1 = [np.asarray(points_plane[i][0]) for i in range(len(points_plane))]
    p2 = [np.asarray(points_plane[i][1]) for i in range(len(points_plane))]
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p1 = np.vstack(p1)
    p2 = np.vstack(p2)

    return p1, p2


def get_structure(match, img, vo):
    # Get structure of the scene
    # @param match: Matcher object
    # @apram img: CVImage object
    # @param vo: VisualOdometry object

    vo.P_from_F(vo.F)
    vo.create_P1()

    # Triangulate points
    # global_kpts1 --> keypoints in the second scene, but we are inverting the
    # scene in order to obtain the movement of the camera, which is equal to the
    # movement of the points inverted. So, the camera matrix of the global_kpts1
    # keypoints is the camera of the first image

    scene = vo.opt_triangulation(match.global_kpts1, match.global_kpts2,
                                 vo.cam1.P, vo.cam2.P)

    return scene


def get_structure_normalized(match, img, vo):
    # Get structure of the scene
    # @param match: Matcher object
    # @apram img: CVImage object
    # @param vo: VisualOdometry object

    vo.create_P1()

    # Triangulate points
    # global_kpts1 --> keypoints in the second scene, but we are inverting the
    # scene in order to obtain the movement of the camera, which is equal to the
    # movement of the points inverted. So, the camera matrix of the global_kpts1
    # keypoints is the camera of the first image

    scene = vo.opt_triangulation(match.global_kpts1, match.global_kpts2,
                                 vo.cam1.P, vo.cam2.P)

    return scene


def plot_plane(X, Y, Z, points):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', s=50)
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis('equal')
    ax.axis('tight')
    plt.show()
