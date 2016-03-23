from VisualOdometry import VisualOdometry
from utils import get_structure, plot_matches, correlate_image, plot_scene
from utils import get_structure_normalized, plot_two_points
from CVImage import CVImage
from Matcher import Matcher
import cv2
import numpy as np
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS |
                                                         cv2.TERM_CRITERIA_COUNT,
                                                         10, 0.03))

def run():
    match = Matcher()
    img = CVImage('/home/cesar/Documentos/Computer_Vision/01/image_0')
    # img = CVImage('/home/cesar/Documentos/Computer_Vision/images_test')

    # Load images
    img.read_image()
    img.copy_image()
    img.acquire()
    # t = threading.Thread(target=plot_image, args=(img.new_image, ))
    # t.start()

    # Correlate

    p1, p2 = correlate_image(match, img, 2, 7)
    print ("Total number of keypoints in second image: \
           {}".format(len(match.global_kpts1)))

    print ("Total number of keypoints in first image: \
           {}".format(len(match.global_kpts2)))

    if not match.is_minmatches:
        print "There aren't matches after filtering. Iterate to next image..."
        return

    # Plot keypoints
    plot_matches(match, img)
    # t.__stop()

    # Now, estimate F
    vo = VisualOdometry()
    match.global_kpts1, match.global_kpts2 = \
        vo.EstimateF_multiprocessing(match.global_kpts2, match.global_kpts1)

    # Get structure of the scene, up to a projectivity
    scene = get_structure(match, img, vo)

    # Optimize F
    # param_opt, param_cov = vo.optimize_F(match.global_kpts1, match.global_kpts2)
    # vo.cam2.set_P(param_opt[:9].reshape((3, 3)))
    # scene = vo.recover_structure(param_opt)

    # Plot it
    plot_scene(scene)

    # Get the Essential matrix
    vo.E_from_F()
    print vo.F
    print vo.E

    # Recover pose
    R, t = vo.get_pose(match.global_kpts1, match.global_kpts2,
                       vo.cam1.focal, vo.cam1.pp)
    print R

    print t

    # Compute camera matrix 2
    print "CAM2", vo.cam2.P
    vo.cam2.compute_P(R, t)
    print "CAM2", vo.cam2.P

    # Get the scene
    scene = get_structure_normalized(match, img, vo)
    plot_scene(scene)

    # What have we stored?
    print ("Permanent Keypoints in the first image stored: \
           {}".format(type(match.curr_kp[0])))
    print ("Permanent descriptors in the first image stored: \
           {}".format(len(match.curr_dsc)))

    print ("Format of global keypoints: \
           {}".format(type(match.global_kpts1)))

    print ("Format of global keypoints: \
            {}".format(type(match.global_kpts1[0])))
    print ("Shape of global kpts1: {}".format(np.shape(match.global_kpts1)))

    # print ("global keypoint: \
    #       {}".format(match.global_kpts1[0]))
    # Acquire image
    img.copy_image()
    img.acquire()
    d, prev_points, points_tracked = match.lktracker(img.prev_image, \
                                                     img.new_image,
                                                     match.global_kpts2)
    print ("Points tracked: \ {}".format(len(points_tracked)))

    plot_two_points(np.reshape(match.global_kpts2,
                               (len(match.global_kpts2), 2)),
                    prev_points, img)
    test = []
    for (x, y), good_flag in zip(match.global_kpts2, d):
        if not good_flag:
            continue
        test.append((x, y))
    # plot_two_points(np.reshape(match.global_kpts2, (len(match.global_kpts2), 2)),
     #                np.asarray(points_tracked), img)
    plot_two_points(np.asarray(test), np.asarray(points_tracked), img)
    # points, st, err = cv2.calcOpticalFlowPyrLK(img.prev_grey, img.new_image,
    #                                            match.global_kpts2, None,
    #                                            **lk_params)
    # print len(points)
    print "Shape of p1: {}".format(np.shape(p1))
    plane = vo.opt_triangulation(p1, p2,
                                 vo.cam1.P, vo.cam2.P)
    plot_scene(plane)
    print "Shpe of plane: {}".format(np.shape(plane))
    print "Type of plane: {}".format(type(plane))
    print np.transpose(plane[:, :3])
    print plane[:, 1]

if __name__ == '__main__':
    run()
