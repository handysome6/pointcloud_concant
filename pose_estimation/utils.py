import cv2
import numpy as np
import time
import copy
import mrcal 
from imgproc.mrcal_rectify import unproject_to_3dcoords, get_camera_fov
from auto_match.aruco import ArucoDetector
from .feature_extract import detectAndDescribe, my_match, opencv_trible_match

class CameraModel():
    def __init__(self, cameramodel_0_path, cameramodel_1_path):
        with open(cameramodel_0_path, "r") as file:
            camera_model_0 = mrcal.cameramodel(cameramodel_0_path)
        with open(cameramodel_1_path, "r") as file:
            camera_model_1 = mrcal.cameramodel(cameramodel_1_path)

        self.getParameters(camera_model_0, camera_model_1)

    def getParameters(self,cameramodel_0,cameramodel_1):
        self.models = [cameramodel_0, cameramodel_1]
        hfov, vfov = get_camera_fov(self.models[0])
        az_fov_deg = int(hfov) + 2
        el_fov_deg = int(vfov) + 2
        MODEL = 'LENSMODEL_PINHOLE'
        self.models_rectified = mrcal.rectified_system(self.models,
                                                az_fov_deg=az_fov_deg,
                                                el_fov_deg=el_fov_deg,
                                                rectification_model=MODEL)
        self.rectification_maps = mrcal.rectification_maps(self.models, self.models_rectified)

        # Rectified left camera intrinsics
        intrinsics_0 = self.models_rectified[0].intrinsics()
        internal_matrix_left = [[intrinsics_0[1][0], 0, intrinsics_0[1][2]],[0, intrinsics_0[1][1], intrinsics_0[1][3]],[0, 0, 1]]
        internal_matrix_left = np.array(internal_matrix_left)
        self.P1 = internal_matrix_left
    
        # Rectified light camera intrinsics
        intrinsics_1 = self.models_rectified[1].intrinsics()
        internal_matrix_right = [[intrinsics_1[1][0], 0, intrinsics_1[1][2]],[0, intrinsics_1[1][1], intrinsics_1[1][3]],[0, 0, 1]]
        internal_matrix_right = np.array(internal_matrix_right)
        self.P2 = internal_matrix_right

        # Raw left camera intrinsics
        intrinsics_0 = self.models[0].intrinsics()
        internal_matrix_left_raw = [[intrinsics_0[1][0], 0, intrinsics_0[1][2]],[0, intrinsics_0[1][1], intrinsics_0[1][3]],[0, 0, 1]]
        internal_matrix_left_raw = np.array(internal_matrix_left_raw)
        self.P1_raw = internal_matrix_left_raw
    
        # Raw light camera intrinsics
        intrinsics_1 = self.models[1].intrinsics()
        internal_matrix_right_raw = [[intrinsics_1[1][0], 0, intrinsics_1[1][2]],[0, intrinsics_1[1][1], intrinsics_1[1][3]],[0, 0, 1]]
        internal_matrix_right_raw = np.array(internal_matrix_right_raw)
        self.P2_raw = internal_matrix_right_raw

        # Rectified extrinsics
        extrinsics_0 = self.models_rectified[0].extrinsics_rt_fromref()
        extrinsics_1 = self.models_rectified[1].extrinsics_rt_fromref()
        self.R_0, _ = cv2.Rodrigues(np.array(extrinsics_0[:3]))
        self.R_1, _ = cv2.Rodrigues(np.array(extrinsics_1[:3]))
        
        self.extrinsics_rect = np.eye(4)
        self.extrinsics_rect[:3,:3] = self.R_1@(np.linalg.inv(self.R_0))
        self.extrinsics_rect[:3, 3] = np.array(extrinsics_1[3:])

        # Raw extrinsics
        extrinsics_0 = self.models[0].extrinsics_rt_fromref()
        extrinsics_1 = self.models[1].extrinsics_rt_fromref()
        self.R_0_raw, _ = cv2.Rodrigues(np.array(extrinsics_0[:3]))
        self.R_1_raw, _ = cv2.Rodrigues(np.array(extrinsics_1[:3]))

        self.extrinsics_raw = np.eye(4)
        self.extrinsics_raw[:3,:3] = self.R_1_raw
        self.extrinsics_raw[:3, 3] = np.array(extrinsics_1[3:])

        R_1_right_coord = self.R_1@(np.linalg.inv(self.R_1_raw))
        self.H_l = internal_matrix_left@self.R_0@np.linalg.inv(internal_matrix_left_raw)
        self.H_r = internal_matrix_right@R_1_right_coord@np.linalg.inv(internal_matrix_right_raw)

    def get_undistorted_image(self, left_image, right_image):
        K_left = self.P1_raw
        dist_left = self.models[0].intrinsics()[1][4:]
        undistorted_left_image = cv2.undistort(left_image, K_left, dist_left)

        K_right = self.P2_raw
        dist_right = self.models[1].intrinsics()[1][4:]
        undistorted_right_image = cv2.undistort(right_image, K_right, dist_right)

        return undistorted_left_image, undistorted_right_image

    def rectify_image(self, image00: str, image01: str):
        images = [mrcal.load_image(f) for f in (image00, image01)]
        images_rectified = [mrcal.transform_image(images[i], self.rectification_maps[i]) for i in range(2)]
        if len(images_rectified[0].shape) == 2:
            images_rectified = [np.dstack([img]*3) for img in images_rectified]

        return images_rectified

    def get3DPoints_in_raw(self, left_points, right_points):
        disp = left_points[:,0] - right_points[:,0]
        points_3d_list = []
        for i in range(len(left_points)):
            p_rect0 = mrcal.stereo_unproject(disp[i], self.models_rectified, disparity_scale = 1, qrect0 = left_points[i].reshape(-1, 2).astype(np.float64))
            Rt_cam0_rect0 = mrcal.compose_Rt(self.models[0].extrinsics_Rt_fromref(),
                                            self.models_rectified[0].extrinsics_Rt_toref() )
            p_cam0 = mrcal.transform_point_Rt(Rt_cam0_rect0, p_rect0)
            points_3d_list.append(p_cam0[0])
        return np.array(points_3d_list).squeeze()

    def get3DPoints(self, left_points, right_points):
        disp = left_points[:,0] - right_points[:,0]
        points_3d_list = []
        for i in range(len(left_points)):
            p_rect0 = mrcal.stereo_unproject(disp[i], self.models_rectified, disparity_scale = 1, qrect0 = left_points[i].reshape(-1, 2).astype(np.float64))
            # Rt_cam0_rect0 = mrcal.compose_Rt(self.models[0].extrinsics_Rt_fromref(),
            #                                 self.models_rectified[0].extrinsics_Rt_toref() )
            # p_cam0 = mrcal.transform_point_Rt(Rt_cam0_rect0, p_rect0)
            points_3d_list.append(p_rect0)
        return np.array(points_3d_list).squeeze()

    def stereo_2_3DPoints(self, left_points, right_points):
        disp = left_points[:,0] - right_points[:,0]
        v_diff = left_points[:,1] - right_points[:,1]
        save_indexes = abs(v_diff) < 5

        in_range_left = left_points[:,0] > 1300
        save_indexes = save_indexes & in_range_left
        in_range_right = right_points[:,0] < 4500
        save_indexes = save_indexes & in_range_right

        points_3d_list = []
        for i in range(len(left_points)):
            if save_indexes[i]:
                points_3d = mrcal.stereo_unproject(disp[i], self.models_rectified, disparity_scale = 1, qrect0 = left_points[i].reshape(-1, 2).astype(np.float64))
                points_3d_list.append(points_3d)
        return np.array(points_3d_list).squeeze(), left_points[save_indexes], right_points[save_indexes], save_indexes

class aruco_det():
    def __init__(self):
        self.left_points = None
        self.right_points = None

    def det_3dpoints(self, left_img, right_img, camera_model):
        # detect aruco marker pairs
        left_ad = ArucoDetector(left_img)
        right_ad = ArucoDetector(right_img)

        self.left_points, self.left_ids = left_ad.get_corners_centers_ids()
        self.right_points, self.right_ids = right_ad.get_corners_centers_ids()

        f1_l_r_matched_f1l_corners, f1_l_r_matched_f1r_corners, f1_l_r_id_matched = match_2frame_aruco_corners(self.left_ids, self.left_points, self.right_ids, self.right_points)
        
        self.left_centers = f1_l_r_matched_f1l_corners[:,-1,:]
        self.right_centers = f1_l_r_matched_f1r_corners[:,-1,:]

        self.left_corners = f1_l_r_matched_f1l_corners[:,:4,:]
        self.right_corners = f1_l_r_matched_f1r_corners[:,:4,:]
        self.matched_ids = f1_l_r_id_matched

        # my_draw_matches(f1_l_r_matched_f1l_corners, f1_l_r_matched_f1r_corners, left_img, right_img)

        # Calculating 3D points
        center_world_points = camera_model.get3DPoints(self.left_centers, self.right_centers)

        corner_world_points = camera_model.get3DPoints(self.left_corners.reshape(-1, 2), self.right_corners.reshape(-1, 2)) # [N*4, 3]

        # center_world_points = camera_model.get3DPoints_in_raw(self.left_centers, self.right_centers)
        # corner_world_points = camera_model.get3DPoints_in_raw(self.left_corners.reshape(-1, 2), self.right_corners.reshape(-1, 2)) # [N*4, 3]

        self.corner_world_points = corner_world_points.reshape(-1, 4, 3) # [N, 4, 3]

        return center_world_points, f1_l_r_id_matched

def match_2frame_aruco_corners(left_id, left_corners, right_id, right_corners):
    '''
    input:
        left_id: N, 
        left_corners: N, 4, 2
    return:
        left_corners_matched_sorted: M, 4, 2
        right_corners_matched_sorted: M, 4, 2
        left_id_matched_sorted: M,
    '''

    common_id = np.intersect1d(left_id, right_id)
    indices_left_id = np.isin(left_id, common_id)
    indices_right_id = np.isin(right_id, common_id)

    left_corners_matched = left_corners[indices_left_id]
    left_id_matched = left_id[indices_left_id]
    left_corners_matched_sorted = left_corners_matched[np.argsort(left_id_matched)]

    right_corners_matched = right_corners[indices_right_id]
    right_id_matched = right_id[indices_right_id]
    right_corners_matched_sorted = right_corners_matched[np.argsort(right_id_matched)]
    left_id_matched_sorted = np.sort(left_id_matched)
    return left_corners_matched_sorted, right_corners_matched_sorted, left_id_matched_sorted


def estimate_from_cv_feat(imageA, imageB, imageC, imageD, camera_model):
    kpsA, featuresA = detectAndDescribe(imageA, method='sift')
    kpsB, featuresB = detectAndDescribe(imageB, method='sift')
    kpsC, featuresC = detectAndDescribe(imageC, method='sift')

    matches_AB = my_match(kpsA, kpsB, featuresA, featuresB, ratio=0.5)
    matches_AC = my_match(kpsA, kpsC, featuresA, featuresC, ratio=0.5)

    frame1_left_points, frame1_right_points, frame2_left_points = opencv_trible_match(kpsA, kpsB, kpsC, matches_AB, matches_AC)
    # No overlap
    if len(frame1_left_points) < 10:
        return None
    dist=np.mat([0,0,0,0,0])
    # world_points = camera_model.get3DPoints(frame1_left_points, frame1_right_points) # world_points in raw coordinate
    world_points, frame1_left_points, frame1_right_points, save_index  = camera_model.stereo_2_3DPoints(frame1_left_points, frame1_right_points)
    frame2_left_points = frame2_left_points[save_index]
    # my_draw_matches(frame1_left_points, frame1_right_points, imageA, imageB)
    # my_draw_matches(frame1_left_points, frame2_left_points, imageA, imageC)
    # _, rvec, tvec = cv2.solvePnP(world_points, frame2_left_points, camera_model.P1, dist)
    # retval, rvec, tvec, inliers = cv2.solvePnPRansac(world_points, frame2_left_points, camera_model.P1, np.array([0.0,0.0,0.0,0.0,0.0]))

    success, rvec, tvec, inliers = cv2.solvePnPRansac(world_points, frame2_left_points, camera_model.P1, np.array([0.0,0.0,0.0,0.0,0.0]))
    # @TODO Use right image to refine
    if success and inliers is not None:
        rvec, tvec = cv2.solvePnPRefineLM(world_points[np.squeeze(inliers)], frame2_left_points[np.squeeze(inliers)], camera_model.P1, dist, rvec, tvec)
        # success, rvec, tvec = cv2.solvePnP(world_points[np.squeeze(inliers)], frame2_left_points[np.squeeze(inliers)],
        #     camera_model.P1, dist, rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)

    rotMat, _ = cv2.Rodrigues(np.array(rvec))
    T_mat_opencv = np.eye(4)
    T_mat_opencv[:3,:3] = rotMat
    T_mat_opencv[:3, 3:] = tvec
    # print("T_mat_opencv: \n", T_mat_opencv)
    return T_mat_opencv


def draw_circle(img, point, point_color=(0, 0, 255)):
    point_size = 1
    # point_color = (0, 0, 255) # BGR
    thickness = 4
    cv2.circle(img, (int(point[0]), int(point[1])), point_size, point_color, thickness)
    cv2.circle(img, (int(point[0]), int(point[1])), 30, point_color, 10)
    return img

def draw_points_reprojection(pixel_left, pixel_right, f1_id, left_points_in_raw, right_points_in_raw, f2_id, undistorted_left_image, undistorted_right_image):
    left_img = copy.copy(undistorted_left_image)
    right_img = copy.copy(undistorted_right_image)
    point_size = 1
    point_color = (0, 0, 255) # BGR
    thickness = 4

    common_id = np.intersect1d(f1_id, f2_id)
    indices_f1_id = np.isin(f1_id, common_id)
    indices_f2_id = np.isin(f2_id, common_id)
    f1_id_matched = f1_id[indices_f1_id]
    f2_id_matched = f2_id[indices_f2_id]

    pixel_left_matched = pixel_left[indices_f1_id]
    pixel_left_matched_sorted = pixel_left_matched[np.argsort(f1_id_matched)]
    pixel_right_matched = pixel_right[indices_f1_id]
    pixel_right_matched_sorted = pixel_right_matched[np.argsort(f1_id_matched)]

    left_points_in_raw_matched = left_points_in_raw[indices_f2_id]
    left_points_in_raw_matched_sorted = left_points_in_raw_matched[np.argsort(f2_id_matched)]
    right_points_in_raw_matched = right_points_in_raw[indices_f2_id]
    right_points_in_raw_matched_sorted = right_points_in_raw_matched[np.argsort(f2_id_matched)]

    # import pdb; pdb.set_trace()

    for i,point in enumerate(pixel_left_matched_sorted):
        # cv2.circle(left_img, (int(point[0]), int(point[1])), point_size, point_color, thickness)
        # cv2.circle(left_img, (int(point[0]), int(point[1])), 30, point_color, 10)
        # cv2.circle(left_img, (int(left_points_in_raw_matched_sorted[i][0]), int(left_points_in_raw_matched_sorted[i][1])), point_size, (0, 255, 0), thickness)
        left_img = draw_circle(left_img, point, (0, 0, 255))
        left_img = draw_circle(left_img, left_points_in_raw_matched_sorted[i], (0, 255, 0))
        distance_error = np.linalg.norm(point - left_points_in_raw_matched_sorted[i])
        text = '{:.4f}'.format(distance_error)
        cv2.putText(left_img, text, (int(point[0])+10, int(point[1])+10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)
    
    for i,point in enumerate(pixel_right_matched_sorted):
        # cv2.circle(right_img, (int(point[0]), int(point[1])), point_size, point_color, thickness)
        # cv2.circle(right_img, (int(right_points_in_raw_matched_sorted[i][0]), int(right_points_in_raw_matched_sorted[i][1])), point_size, (0, 255, 0), thickness)
        right_img = draw_circle(right_img, point, (0, 0, 255))
        right_img = draw_circle(right_img, right_points_in_raw_matched_sorted[i], (0, 255, 0))

        distance_error = np.linalg.norm(point - right_points_in_raw_matched_sorted[i])
        text = '{:.4f}'.format(distance_error)
        cv2.putText(right_img, text, (int(point[0])+10, int(point[1])+10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)    
    
    show_img = cv2.hconcat([left_img, right_img])#水平拼接
    return show_img

def draw_2sets_points(pixel_left, pixel_right, input_left_img, input_right_img, window_name="Show Points"):
    point_size = 1
    point_color = (0, 0, 255) # BGR
    thickness = 4 # 0、4、8
    left_img = copy.copy(input_left_img)
    right_img = copy.copy(input_right_img)

    for i,point in enumerate(pixel_left):
        cv2.circle(left_img, (int(point[0]), int(point[1])), point_size, point_color, thickness)
        cv2.circle(left_img, (int(point[0]), int(point[1])), 30, point_color, 10)
    point_color = (255, 0, 0)
    for i,point in enumerate(pixel_right):
        cv2.circle(right_img, (int(point[0]), int(point[1])), point_size, point_color, thickness)
        cv2.circle(right_img, (int(point[0]), int(point[1])), 30, point_color, 10)

    show_img = cv2.hconcat([left_img, right_img])#水平拼接
    cv2.namedWindow(window_name, 0)
    cv2.resizeWindow(window_name, 1200, 800)
    cv2.imshow(window_name, show_img)
    cv2.waitKey (0)
    cv2.destroyAllWindows()

def my_draw_matches(left_kpt, right_kpt, left_img, right_img, window_name="Matched"):
    kp1 = []
    kp2 = []
    goodMatch = []
    for i in range(left_kpt.shape[0]):
        kp1.append(cv2.KeyPoint(left_kpt[i,0], left_kpt[i,1], 2))
        kp2.append(cv2.KeyPoint(right_kpt[i,0], right_kpt[i,1], 2))
        goodMatch.append(cv2.DMatch(i, i, 1))
    img_show = cv2.drawMatches(left_img, kp1, right_img, kp2, goodMatch, None, flags=2)
    cv2.namedWindow(window_name, 0)
    cv2.resizeWindow(window_name, 1400, 600)  # 设置窗口大小
    cv2.imshow(window_name, img_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_undistorted_image(left_image, right_image, models):
    intrinsics_0 = models[0].intrinsics()
    P1_raw = [[intrinsics_0[1][0], 0, intrinsics_0[1][2]],[0, intrinsics_0[1][1], intrinsics_0[1][3]],[0, 0, 1]]
    P1_raw = np.array(P1_raw)

    intrinsics_1 = models[1].intrinsics()
    P2_raw = [[intrinsics_1[1][0], 0, intrinsics_1[1][2]],[0, intrinsics_1[1][1], intrinsics_1[1][3]],[0, 0, 1]]
    P2_raw = np.array(P2_raw)

    K_left = P1_raw
    dist_left = models[0].intrinsics()[1][4:]
    undistorted_left_image = cv2.undistort(left_image, K_left, dist_left)

    K_right = P2_raw
    dist_right = models[1].intrinsics()[1][4:]
    undistorted_right_image = cv2.undistort(right_image, K_right, dist_right)

    return undistorted_left_image, undistorted_right_image


def project3DPoints(world_points, P1, P2, external_matrix_right):
    '''
        world_points: 3D Points in rectified left camera coordinate
    '''
    point_cloud_right_camera = np.dot(external_matrix_right[:3,:3], world_points.T) + external_matrix_right[:3, [3]]
    pixel_right = np.dot(P2, point_cloud_right_camera)
    pixel_right = pixel_right[:2] / pixel_right[2]
    pixel_right = pixel_right.T

    pixel_left = world_points@P1.T
    pixel_left = pixel_left[:, :2] / pixel_left[:, [2]]

    return pixel_left, pixel_right