# Computer_Vision
Source code for my projects of Computer Vision (outside of ROS)

TODO: a) Function to measure the mean motion (in pixels) between the two first frames, so I can stablish the number of rois.
      b) Function to extract the pose from the fundamental matrix (via essential matrix).
      c) Check the data that have to be stored between frames.
      d) Function that associates each feature point (vector (x, y)) to each camera. Maybe a 3D matrix would be the solution.
      e) The same for the 3D points (structure of the scene)
      f) Function to fit a plane to the 3D point cloud.
      g) Function to get the scale of the scene.
      h) Function that determines when to create a new keyframe.
      i) Function to search in the neighborhood of the keypoints in the next frame (tracking the keypoints).
      j) Sparsify the Levenberg-Marquardt algorithm, at least the Jacobian of the reprojection error.
      k) Investigate the use of the convex optimization approach (CVXOPT library)
