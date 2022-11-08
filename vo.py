import numpy as np
import cv2
from tqdm import tqdm
import json
from matplotlib import pyplot as plt
from lib.visualization import plotting

from skimage.measure import ransac
from skimage.transform import AffineTransform

class VisualOdometry():
    def __init__(self, video_dir="", calibration_dir="", live_data=False, undistort=False):
        self.Dist, self.K, self.P = self._load_calib_from_json(calibration_dir)
        if not live_data:
            self.images = self._load_images_from_video(video_dir, undistort)
        else:
            self.images = []
        self.MIN_MATCH_COUNT = 10
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)



    def _load_images_from_video(self, filepath, undistort):
        imageFrames = []
        try:
            vidcap = cv2.VideoCapture(filepath)
        except:
            print(f"Cannot read file {filepath}. Returning empty.")
            return imageFrames
        count = 0
        success, image = vidcap.read()
        while success:
            if undistort:
                h, w, _ = image.shape
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.K, self.Dist, (w,h), 1, (w,h))
                image = cv2.undistort(image, self.K, self.Dist, None, newcameramtx)
            imageFrames.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            success, image = vidcap.read()
            count += 1

        return imageFrames


    @staticmethod
    def _load_calib_from_json(filepath):
        f = open(filepath)
        jsonData = json.load(f)

        Dist = np.array(jsonData["Dist"])
        K = np.array(jsonData["Matrix"])
        P = np.concatenate((K, np.array([[0],[0],[0]])), axis=1)

        return Dist, K, P

    def get_matches(self, i):


        # Find the keypoints and descriptors with ORB
        img1 = self.images[i-1]
        img2 = self.images[i]

        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)
        # Find matches
        matches = self.flann.knnMatch(des1, des2, k=2)

        # Find the matches there do not have a to high distance
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        self.numGood = len(good)

        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 2)

        if self.numGood > self.MIN_MATCH_COUNT:
            # Ransac
            model, inliers = ransac(
                    (src_pts, dst_pts),
                    AffineTransform, min_samples=4,
                    residual_threshold=8, max_trials=100
                ) #10, 8, 100, gives decent results maybe

            n_inliers = np.sum(inliers)

            inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
            inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
            placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
            image3 = cv2.drawMatches(img1, inlier_keypoints_left, img2, inlier_keypoints_right, placeholder_matches, None)

            cv2.imshow('Matches', image3)
            cv2.waitKey(5)

            src_pts = np.float32([ inlier_keypoints_left[m.queryIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
            dst_pts = np.float32([ inlier_keypoints_right[m.trainIdx].pt for m in placeholder_matches ]).reshape(-1, 2)

        return src_pts, dst_pts

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_pose(self, q1, q2):
        """
        Calculates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix
        """

        if self.numGood > self.MIN_MATCH_COUNT:
            # Essential matrix
            E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold=1)
            
            if E is None:
                print(f"Essential matrix returned None.")
                return np.array([])



            # Decompose the Essential matrix into R and t
            R, t = self.decomp_essential_mat(E, q1, q2)

            # Get transformation matrix
            transformation_matrix = self._form_transf(R, np.squeeze(t))
            return transformation_matrix
        else:
            print(f"Number of found good matches is below minimum required: {self.MIN_MATCH_COUNT}. Skipping frame.")
            return np.array([])
    def decomp_essential_mat(self, E, q1, q2):
        """
        Decompose the Essential matrix

        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """
        def sum_z_cal_relative_scale(R, t):
            # Get the transformation matrix
            T = self._form_transf(R, t)
            # Make the projection matrix
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

            # Triangulate the 3D points
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            # Also seen from cam 2
            hom_Q2 = np.matmul(T, hom_Q1)

            # Un-homogenize
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # Find the number of points there has positive z coordinate in both cameras
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

            # Form point pairs and calculate the relative scale
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                     np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        # Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        # Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        return [R1, t]

def main():
    video_dir = "data\\smallLIndoors\\small_L_indoors.avi"
    calibration_dir = "calibration.json"
    live_data = False
    vo = VisualOdometry(video_dir=video_dir, calibration_dir=calibration_dir, live_data=live_data, undistort=True)
    
    if live_data:
        cap = cv2.VideoCapture()
        # The device number might be 0 or 1 depending on the device and the webcam
        cap.open(1, cv2.CAP_DSHOW)
        cur_pose = np.eye(4)
        frameNum = 0

        xs = []
        ys = []

        while True:
            try:
                ret, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if frameNum == 0 or frameNum == 1:
                    vo.images.append(frame)
                    frameNum += 1
                    continue
                else:
                    vo.images[0] = vo.images[1]
                    vo.images[1] = frame

                q1, q2 = vo.get_matches(1)
                transform = vo.get_pose(q1, q2)
                if transform.size == 0:
                    continue
                cur_pose = np.matmul(cur_pose, np.linalg.inv(transform))
                x = cur_pose[0, 3]
                y = cur_pose[2, 3]

                xs.append(x)
                ys.append(y)
                frameNum += 1
            except:
                break
            """ cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break """
            
        cap.release()
        cv2.destroyAllWindows()
        t = np.arange(len(xs))
        plt.scatter(xs,ys, c=t)
        plt.ylim(min(ys)-5, max(ys)+5)
        plt.xlim(min(xs)-5, max(xs)+5)
        plt.show()
    else:
        f = open("data\\smallLIndoors\\VideoTimestamps.json", "r")
        timestampsStuct = json.load(f)
        timestamps = timestampsStuct["TimeStamps"]
        poses = {}
        estimated_path = []
        gt_path = []
        cur_pose = np.eye(4)
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
        gt_path.append((cur_pose[0, 3], cur_pose[2, 3]))
        for i in tqdm(range(1,len(vo.images))):
            q1, q2 = vo.get_matches(i)
            transform = vo.get_pose(q1, q2)
            if transform.size == 0:
                continue
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transform))
            dataStruct = {
                "pos": [cur_pose[0, 3], cur_pose[2, 3]],
                "timestamp" : timestamps[i],
            }
            poses[i] = dataStruct
            gt_path.append((cur_pose[0, 3], cur_pose[2, 3]))
            estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
        plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry", file_out="Plotting.html")
        with open("poses.json", "w") as f:
            json.dump(poses,f)



if __name__ == "__main__":
    main()