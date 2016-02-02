import numpy as np
import cv2
import threading
from multiprocessing import Process, Queue
from scipy import linalg


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
        self.mask = None
        self.H = None
        self.maskH = None
        self.e = None  # epipole
        self.P1 = None  # First camera matrix
        self.P2 = None  # Second camera matrix
        self.correctedkpts1 = None
        self.correctedkpts2 = None  # Feature locations corrected by the optimal
        # triangulation algorithm

        self.structure = None  # List  of 3D points (triangulated)

    def FindFundamentalRansac(self, kpts1, kpts2):
        # Compute Fundamental matrix from a set of corresponding keypoints,
        # within a RANSAC scheme
        # @param kpts1: list of keypoints of the current frame
        # @param kpts2: list of keypoints of the reference frame

        kpts1 = np.float32(kpts1)
        kpts2 = np.float32(kpts2)
        self.F, self.mask = cv2.findFundamentalMat(kpts1, kpts2, cv2.FM_RANSAC)

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

        print F

        U, S, V = linalg.svd(F)
        print V
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
        self.P2 = np.vstack((np.dot(Te, F.T).T, self.e)).T

    def create_P1(self):
        # Initialize P1 = [I | 0]

        self.P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

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

        points3D = cv2.triangulatePoints(self.P1, self.P2, kpts1, kpts2)

        X = points3D / points3D[3]  # Normalize points: [x, y, 1]

        # The individual points are selected like these:

        # print points3D[:, 0]  # First point, first column

        # print "X", X[:, 0]

        # points2d = np.float32(np.dot(self.P2, X))

        # points2d = self.project_3dpoints(X, self.P2)

        # Euclidean coordinates.

        # points2d = self.convert_from_homogeneous(points2d)

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

        for i in range(len(kpts)):

            kpts[i].reshape(-1, 1, 3)

            kpts[i] = np.array(kpts[i], np.float32).reshape(-1, 1, 3)


        pnh = [cv2.convertPointsFromHomogeneous(x) for x in kpts]

        return pnh

    def project_3dpoints(self, points3d, P):
        # Project 3d points in the image plane via the projection matrix P
        # @param points3d: list of 3d points (normalized homogeneous)
        # @param P: Projection matrix (homogeneous)
        # return 2dpoints: list of 2d points in homogeneous coordinates

        points3d = points3d.T

        listpoints3d = []

        print len(points3d[:])

        for i in range(len(points3d)):

            listpoints3d.append(points3d[i])


        # points3d.tolist()

        listpoints2d = []

        for i in range(len(listpoints3d)):

            listpoints2d.append(np.float32(np.dot(P, listpoints3d[i].T)))

        # points2d = [np.float32(np.dot(P, x.T)) for x in points3d]

        return listpoints2d
