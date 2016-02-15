import numpy as np
import cv2
import threading
from multiprocessing import Process, Queue
from scipy import linalg
from scipy import optimize
from scipy.spatial.distance import cdist

# @file VisualOdometry.py
# @author Cesar
# @version 1.0
# Class VisualOdometry. This class contain the measured
# odometry. Implements the calculus of the odometry based on two consecutive
# images.


class VisualOdometry(object):
    def __init__(self):
        self.F = None
        self.inlier_points_new = None
        self.inlier_points_prev = None
        self.outlier_points_new = None
        self.outlier_points_prev = None
        self.mask = None
        self.H = None
        self.maskH = None
        self.e = None  # epipole
        self.cam1 = Camera(None)
        self.cam2 = Camera(None)
        self.correctedkpts1 = None
        self.correctedkpts2 = None  # Feature locations corrected by the optimal
        # triangulation algorithm. This are the estimated features in the Gold
        # Standard algorithm for estimating F (H&Z page 285)
        self.K = np.array([[7.18856e+02, 0.0, 6.071928e+02],
                          [0.0, 7.18856e+02, 1.852157e+02],
                          [0.0, 0.0, 1.0]])  # Calibration matrix
        self.E = None  # Essential matrix
        self.maskE = None  # Mask for the essential matrix

        self.structure = None  # List  of 3D points (triangulated)

    def FindFundamentalRansac(self, kpts1, kpts2):
        # Compute Fundamental matrix from a set of corresponding keypoints,
        # within a RANSAC scheme
        # @param kpts1: list of keypoints of the current frame
        # @param kpts2: list of keypoints of the reference frame

        kpts1 = np.float32(kpts1)
        kpts2 = np.float32(kpts2)
        self.F, self.mask = cv2.findFundamentalMat(kpts1, kpts2, cv2.FM_RANSAC)

    def FindEssentialRansac(self, kpts1, kpts2):
        # Compute Essential matrix from a set of corresponding points
        # @param kpts1: list of keypoints of the current frame
        # @param kpts2: list of keypoints of the reference frame

        kpts1 = np.float32(kpts1)
        kpts2 = np.float32(kpts2)

        # findEssentialMat takes as arguments, apart from the keypoints of both
        # images, the focal length and the principal point. Looking at the
        # source code of this function
        # (https://github.com/Itseez/opencv/blob/master/modules/calib3d/src/five-point.cpp)
        # I realized that these parameters are feeded to the function because it
        # internally create the camera matrix, so they must be in pixel
        # coordinates. Hence, we take them from the already known camera matrix:

        focal = 3.37
        pp = (2.85738, 0.8681)

        # pp = (self.K[0][2], self.K[1][2])

        self.E, self.maskE = cv2.findEssentialMat(kpts2, kpts1, focal, pp,
                                                  cv2.RANSAC, 0.999, 1.0, self.maskE)

    def FindFundamentalRansacPro(self, queue):
        # Compute Fundamental matrix from a set of corresponding keypoints,
        # within a RANSAC scheme
        # @param kpts1: list of keypoints of the current frame
        # @param kpts2: list of keypoints of the reference frame

        temp = queue.get()
        kpts1 = temp[0]
        kpts2 = temp[1]
        F = temp[2]

        F, mask = cv2.findFundamentalMat(kpts1, kpts2, cv2.FM_RANSAC)
        res = [F, mask]
        queue.put(res)

    def EstimateF_multiprocessing(self, kpts1, kpts2):
        # Estimate F using the multiprocessing module
        # @param kpts1: list of keypoints of the current frame
        # @param kpts2: list of keypoints of the reference frame
        # @return kpts1, kpts2: Inliers

        kpts1 = np.float32(kpts1)
        kpts2 = np.float32(kpts2)

        data_queued = [kpts1, kpts2, self.F]
        q = Queue()
        q.put(data_queued)
        # Compute F in a parallel process:

        p = Process(target=self.FindFundamentalRansacPro,
                    args=(q, ))
        p.start()
        # We must wait until the process has finished because then the queue
        # will be filled with the result. Otherwise, we can't use the get method
        # (FIFO model of queue)

        p.join()
        res = q.get()
        self.F = res[0]
        self.mask = res[1]

        # Select only inlier points

        self.outlier_points_new = [kpts1[self.mask.ravel() == 0]]

        self.outlier_points_prev = [kpts2[self.mask.ravel() == 0]]

        self.outlier_points_new = np.float32(self.outlier_points_new[0])

        self.outlier_points_prev = np.float32(self.outlier_points_prev[0])


        return [kpts1[self.mask.ravel() == 1],
                kpts2[self.mask.ravel() == 1]]

    def EstimateF_multithreading(self, kpts1, kpts2):
        t = threading.Thread(target=self.FindFundamentalRansac,
                             args=(kpts2, kpts1, ))
        t.start()

    def FindHomographyRansac(self, kpts1, kpts2):
        # Find the homography between two images given corresponding points
        kpts1 = np.float32(kpts1)
        kpts2 = np.float32(kpts2)

        self.H, self.maskH = cv2.findHomography(kpts1, kpts2, cv2.RANSAC, 1.0)

    def get_epipole(self, F):
        # Return the (right) epipole from a fundamental matrix F.
        # Use with F.T for left epipole
        # @param F: Fundamental Matrix (numpy 3x3 array)

        # Null space of F (Fx = 0)

        U, S, V = linalg.svd(F)
        self.e = V[-1]
        self.e = self.e / self.e[2]

    def skew(self, a):
        # Return the matrix A such that a x v = Av for any v
        # @param a: numpy vector

        return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])

    def P_from_F(self, F):
        # Computes the second camera matrix (assuming P1 = [I 0] from a
        # fundamental matrix.
        # @param F: Numpy matrix (Fundamental)

        F = self.F.T

        self.get_epipole(F)  # Left epipole

        Te = self.skew(self.e)
        self.cam2.set_P(np.vstack((np.dot(Te, F.T).T, self.e)).T)

    def create_P1(self):
        # Initialize P1 = [I | 0]

        self.cam1.set_P(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))

    def optimal_triangulation(self, kpts1, kpts2):
        # For each given point corresondence points1[i] <-> points2[i], and a
        # fundamental matrix F, computes the corrected correspondences
        # new_points1[i] <-> new_points2[i] that minimize the geometric error
        # d(points1[i], new_points1[i])^2 + d(points2[i], new_points2[i])^2,
        # subject to the epipolar constraint new_points2^t * F * new_points1 = 0
        # Here we are using the OpenCV's function CorrectMatches.

        # @param kpts1 : keypoints in one image
        # @param kpts2 : keypoints in the other image
        # @return new_points1 : the optimized points1
        # @return new_points2 : the optimized points2

        # First, we have to reshape the keypoints. They must be a 1 x n x 2
        # array.

        kpts1 = np.float32(kpts1)
        kpts2 = np.float32(kpts2)

        # 3D Matrix : [kpts1[0] kpts[1]... kpts[n]]

        pt1 = np.reshape(kpts1, (1, len(kpts1), 2))

        pt2 = np.reshape(kpts2, (1, len(kpts2), 2))

        new_points1, new_points2 = cv2.correctMatches(self.F, pt1, pt2)

        self.correctedkpts1 = new_points1
        self.correctedkpts2 = new_points2

        # Transform to a 2D Matrix: 2xn

        kpts1 = (np.reshape(new_points1, (len(kpts1), 2))).T
        kpts2 = (np.reshape(new_points2, (len(kpts2), 2))).T

        points3D = cv2.triangulatePoints(self.cam1.P, self.cam2.P, kpts1, kpts2)

        self.structure = points3D / points3D[3]  # Normalize points [x, y, z, 1]

        array = np.zeros((4, len(self.structure[0])))

        for i in range(len(self.structure[0])):

            array[:, i] = self.structure[:, i]

        self.structure = array

        # The individual points are selected like these:

        # self.structure[:, i]. It's a 4 x n matrix

    def triangulate(self, kpts1, kpts2):

        kpts1 = (np.reshape(kpts1, (len(kpts1), 2))).T
        kpts2 = (np.reshape(kpts2, (len(kpts2), 2))).T

        points3D = None

        points3D = cv2.triangulatePoints(self.cam1.P, self.cam2.P, kpts1, kpts2)

        points3D = points3D / points3D[3]

        return points3D

    def convert_from_homogeneous(self, kpts):
        # Convert homogeneous points to euclidean points
        # @param kpts: List of homogeneous points
        # @return pnh: list of euclidean points

        # Remember that every function in OpenCV need us to specify the data
        # type. In addition, convertPointsFromHomogeneous needs the shape of the
        # arrays to be correct. The function takes a vector of points in c++
        # (ie. a list of several points), so in numpy we need a multidimensional
        # array: a x b x c where a is the number of points, b=1, and c=2 to
        # represent 1x2 point data.

        if len(kpts[0]) == 3:

            for i in range(len(kpts)):

                kpts[i].reshape(-1, 1, 3)

                kpts[i] = np.array(kpts[i], np.float32).reshape(-1, 1, 3)

            pnh = [cv2.convertPointsFromHomogeneous(x) for x in kpts]

            for i in range(len(pnh)):

                pnh = np.array(pnh[i], np.float32).reshape(1, 2, 1)

        elif len(kpts[0]) == 4:

            for i in range(len(kpts)):

                kpts[i].reshape(-1, 1, 4)

                kpts[i] = np.array(kpts[i], np.float32).reshape(-1, 1, 4)

            pnh = [cv2.convertPointsFromHomogeneous(x) for x in kpts]

            for i in range(len(pnh)):

                pnh[i] = np.array(pnh[i], np.float32).reshape(1, 3, 1)

        elif len(kpts) == 3:

            pnh = np.zeros((2, len(kpts[0])))

            for i in range(len(kpts[0])):

                pnh[:, i] = kpts[:2, i]

        return pnh

    def convert_array2d(self, kpts):

        print len(kpts[:, 0])

        a = np.zeros((len(kpts[:, 0]), 2))

        for i in range(len(kpts[:, 0])):

            a[i, :] = kpts[i, :2]

        return a

    def project_3dpoints(self, points3d, P):
        # Project 3d points in the image plane via the projection matrix P
        # @param points3d: list of 3d points (normalized homogeneous)
        # @param P: Projection matrix (homogeneous)
        # return 2dpoints: list of 2d points in homogeneous coordinates

        points3d = points3d.T

        listpoints3d = []

        for i in range(len(points3d)):

            listpoints3d.append(points3d[i])

        # points3d.tolist()

        listpoints2d = []

        for i in range(len(listpoints3d)):

            listpoints2d.append(np.float32(np.dot(P, listpoints3d[i].T)))

        # points2d = [np.float32(np.dot(P, x.T)) for x in points3d]

        return listpoints2d

    def project(self, cam,  X):
        # Project points in X (4*n) array and normalize
        # @param cam: camera object
        # @param X: 4*n array of 3d coordinates (homogeneous)

        x = cam.project(X)

        return x

    def functiontominimize(self, params, kpts1, kpts2, structure):
        # This is the function that we will pass to the levenberg-marquardt
        # algorithm. It extracts the camera matrix P' from the params vector
        # (params[0:8]). Then, it triangulate the 3D points
        # X_hat and obtain the estimated x and x'. Finally, it returns the
        # reprojection error (not squared).
        # @param kpts1: inliers in the previous image.
        # @param kpts2: inliers in the second image.
        # @return e_rep: reprojection error
        self.create_P1()
        P2 = None
        P2 = params[0:8].reshape(3, 3)
        # Reshape from (1, nkeypoints, 2) to (nkeypoints, 2)
        # a, b, c = np.shape(self.correctedkpts1)
        # self.correctedkpts1 = self.correctedkpts1.reshape(b, 2)
        # self.correctedkpts2 = self.correctedkpts2.reshape(b, 2)

        # Project the structure to both images and find residual:
        self.cam2.set_P(P2)
        image1 = cam1.project(structure)
        image2 = cam2.project(structure)
        image1 = np.transpose(self.convert_from_homogeneous(image1))
        image2 = np.transpose(self.convert_from_homogeneous(image2))
        error_image1 = self.residual(kpts1, image1)
        error_image2 = self.residual(kpts2, image2)
        error = np.add(error_image1, error_image2)

        return error



        # print type(np.sqrt(error))

        # print "shape", np.shape(np.sqrt(error))

        # print np.sum(error)

        return np.sqrt(error)

    def residual(self, kpts1, kpts2):
        # Reprojection error
        # @param kpts1: numpy nx2 array
        # @param kpts2: numpy nx2 array
        # @return euclidean distance

        # check shape
        if np.shape(kpts1) != np.shape(kpts2):
            kpts2 = np.transpose(kpts2)
        # error = None
        # error_avg = None
        # error = np.subtract(kpts1, kpts2
        # The function cdist returns an array with the distance between every
        # points, and we are interested in the diagonal elements. See
        # documentation.
        return np.diag(cdist(kpts1, kpts2, "sqeuclidean"))

    def minimize_cost(self, kpts1, kpts2):
        # Wrapper for the optimize.leastsq function.
        # First, estimate P2 from F

        params_opt, params_cov = optimize.leastsq(self.functiontominimize,
                                        F0, args=(kpts1, kpts2))

        self.F = np.reshape(F_opt, (3, 3))

    def optimize_F(self, kpts1, kpts2):
        # Wrapper for the optimize.leastsq function.

        # Transform camera matrices into vectors:
        vec_P1 = None
        vec_P2 = None
        vec_P1 = np.hstack(self.cam1.P)
        vec_P2 = np.hstack(self.cam2.P)

        # Transform the structure (matrix 3 x n) to 1d vector
        vec_str = None
        vec_str = self.structure.reshape(-1)

        # Pass them, and additional arguments to leastsq function.
        # TODO: redefine error function

        F_opt, F_cov = optimize.leastsq(self.functiontominimize,
                                        F0, args=(kpts1, kpts2))

        self.F = np.reshape(F_opt, (3, 3))
    def functiontominimize2(self, F, kpts1, kpts2):
        # This is the function that we will pass to the levenberg-marquardt
        # algorithm. It calculate the camera matrices P and P', the last one
        # using the fundamental matrix F. Then, it triangulate the 3D points
        # X_hat and obtain the estimated x and x'. Finally, it returns the
        # reprojection error (not squared).
        # @param kpts1: inliers in the previous image.
        # @param kpts2: inliers in the second image.
        # @return e_rep: reprojection error
        self.create_P1()

        self.optimal_triangulation(kpts1, kpts2)
        # Reshape from (1, nkeypoints, 2) to (nkeypoints, 2)
        a, b, c = np.shape(self.correctedkpts1)
        self.correctedkpts1 = self.correctedkpts1.reshape(b, 2)
        self.correctedkpts2 = self.correctedkpts2.reshape(b, 2)

        A = np.subtract(kpts1, self.correctedkpts1)
        B = np.subtract(kpts2, self.correctedkpts2)

        A = (A**2)
        B = (B**2)

        errora = np.sum(A, axis=1)

        errorb = np.sum(B, axis=1)

        error = np.add(errora, errorb)

        # print type(np.sqrt(error))

        # print "shape", np.shape(np.sqrt(error))

        # print np.sum(error)

        return np.sqrt(error)

    def minimize_cost2(self, kpts1, kpts2):
        # Wrapper for the optimize.leastsq function.

        self.P_from_F(self.F)

        P0 = self.cam2.P
        P_opt, P_cov = optimize.leastsq(self.functiontominimize2,
                                        P0, args=(kpts1, kpts2))

        self.cam2.P = P_opt

    def E_from_F(self):
        #
        # Get the essential matrix from the fundamental
        print np.shape(self.K.transpose())
        print np.shape(self.F)
        self.E = self.K.transpose().dot(self.F).dot(self.K)


class Camera(object):

    def __init__(self, P):
        self.P = P
        self.K = None
        self.R = None
        self.t = None
        self.c = None  # Camera center

    def project(self, X):

        x = np.dot(self.P, X)

        for i in range(3):

            x[i] /= x[2]

        return x

    def set_P(self, P):
        # Set camera matrix
        self.P = P

    def set_K(self, K):

        self.K = K
