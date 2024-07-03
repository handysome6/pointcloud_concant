import open3d as o3d
import numpy as np
import copy
import cv2
from pathlib import Path
from icecream import ic
from typing_extensions import Self

from utils import bilinear_interpolation, timeit
from pose_estimation.feature_extract import detectAndDescribe, my_match


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.2459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

def draw_registration_result(source, target, transformation, color=False):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if not color:
        source_temp.paint_uniform_color([1, 1, 0])
        target_temp.paint_uniform_color([0, 1, 1])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

@timeit
def colored_icp_registration(source, target, voxel_size, current_transformation):
    print("Colored ICP registration")
    voxel_radius = [5*voxel_size, 3*voxel_size, voxel_size]
    max_iter = [60, 35, 20]
    for scale in range(3):
        max_it = max_iter[scale]
        radius = voxel_radius[scale]
        print("scale_level = {0}, voxel_size = {1}, max_iter = {2}".format(scale, radius, max_it))
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)
        source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius*2, max_nn=20))
        target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius*2, max_nn=20))
        result = o3d.pipelines.registration.registration_colored_icp(
            source_down, 
            target_down, 
            radius, 
            current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                              relative_rmse=1e-6,
                                                              max_iteration=max_it))
        current_transformation = result.transformation
        print(result)
    current_transformation = np.array(current_transformation)
    print(current_transformation)
    # draw_registration_result(source, target, current_transformation, color=False)
    return current_transformation


class MyPCD():
    def __init__(self, folder_path):
        self.folder_path = Path(folder_path)
        pcd_path = self.folder_path / 'PointCloud.ply'
        self.pcd = o3d.io.read_point_cloud(str(pcd_path))

        self.image = cv2.imread(str(self.folder_path / 'Image.png'))
        self.h, self.w = self.image.shape[:2]

    def _get_3d_point_int(self, x: int, y: int):
        l = list(self.pcd.points[y * self.w + x])
        return np.array(l)

    def get_3dcoord_bilinear(self, x:float, y:float):
        x0, y0 = int(x), int(y)
        x1, y1 = x0 + 1, y0 + 1

        q00 = self._get_3d_point_int(x0, y0)
        q01 = self._get_3d_point_int(x0, y1)
        q10 = self._get_3d_point_int(x1, y0)
        q11 = self._get_3d_point_int(x1, y1)

        d3_coord = bilinear_interpolation(x, y, x0, y0, x1, y1, q00, q01, q10, q11)
        if np.isnan(d3_coord).any():
            return None
        return d3_coord
    
    def estimate_RT_pnp(self, pcd2: Self, cm: np.ndarray):
        image0 = self.image
        pc0 = self.pcd
        image1 = pcd2.image
        pc1 = pcd2.pcd
        cm = cm

        kps0, features0 = detectAndDescribe(image0, method='sift')
        kps1, features1 = detectAndDescribe(image1, method='sift')
        matches01 = my_match(kps0, kps1, features0, features1, ratio=0.5)

        frame0_points = []
        frame1_points = []
        for m in matches01:
            frame0_points.append([kps0[m.queryIdx].pt[0], kps0[m.queryIdx].pt[1]])
            frame1_points.append([kps1[m.trainIdx].pt[0], kps1[m.trainIdx].pt[1]])

        frame0_points = np.array(frame0_points)
        frame1_points = np.array(frame1_points)
        # No overlap
        if len(frame0_points) < 10:
            return None
        
        dist=np.mat([0,0,0,0,0])
        world_points = []
        delete_ids = []
        for id, (x, y) in enumerate(frame0_points):
            f0_world = self.get_3dcoord_bilinear(x, y)
            if f0_world is not None:
                world_points.append(f0_world)
            else:
                delete_ids.append(id)
        frame0_points = np.delete(frame0_points, delete_ids, axis=0)
        frame1_points = np.delete(frame1_points, delete_ids, axis=0)

        world_points = np.array(world_points)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(world_points, frame1_points, cm, np.array([0.0,0.0,0.0,0.0,0.0]))
        # Use right image to refine
        if success and inliers is not None:
            rvec, tvec = cv2.solvePnPRefineLM(world_points[np.squeeze(inliers)], frame1_points[np.squeeze(inliers)], cm, dist, rvec, tvec)

        rotMat, _ = cv2.Rodrigues(np.array(rvec))
        T_mat_opencv = np.eye(4)
        T_mat_opencv[:3,:3] = rotMat
        T_mat_opencv[:3, 3:] = tvec
        # print("T_mat_opencv: \n", T_mat_opencv)
        return T_mat_opencv

    def remove_plane(self):
        pcd = self.pcd
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                ransac_n=3,
                                                num_iterations=1000)

        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        return outlier_cloud

    def refine_RT_icp(self, pcd2: Self, T_mat_opencv):
        # source = self.remove_plane()
        # target = pcd2.remove_plane()
        source = self.pcd
        target = pcd2.pcd
        threshold = 2
        trans_init = T_mat_opencv

        # downsample the point cloud
        # source = source.voxel_down_sample(voxel_size=0.001)
        # target = target.voxel_down_sample(voxel_size=0.001)

        print("Initial alignment")
        evaluation = o3d.pipelines.registration.evaluate_registration(
            source, target, threshold, trans_init)
        print(evaluation)
        print(trans_init)
        print()
        # visualize the registration result
        # draw_registration_result(source, target, trans_init)


        # print("Recompute the normal of the downsampled point cloud")
        # source.estimate_normals(
        #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # target.estimate_normals(
        #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        print("Apply point-to-point ICP")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint())
            # criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        print()
        print()
        # visualize the registration result
        # draw_registration_result(source, target, reg_p2p.transformation)
        return reg_p2p.transformation

    def estimate_RT_aruco_icp(self, pcd2: Self):
        frame0 = self
        frame1 = pcd2

        # bitwise reverse the image
        image0 = cv2.bitwise_not(frame0.image)
        image1 = cv2.bitwise_not(frame1.image)

        # initialize aruco detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        detecor = cv2.aruco.ArucoDetector(aruco_dict)

        # detect aruco in two images
        corners0, ids0, _ = detecor.detectMarkers(image0)
        corners1, ids1, _ = detecor.detectMarkers(image1)
        dict0 = dict(zip(np.squeeze(ids0), corners0))
        dict1 = dict(zip(np.squeeze(ids1), corners1))

        # get intersection of ids to get common aruco corners
        common_ids = set(ids0.flatten()).intersection(set(ids1.flatten()))
        common_corners0 = np.array([dict0[id] for id in common_ids]).reshape(-1, 2)
        common_corners1 = np.array([dict1[id] for id in common_ids]).reshape(-1, 2)

        # get 3d points from aruco corners
        world_points0 = []
        world_points1 = []
        for corner0, corner1 in zip(common_corners0, common_corners1):
            x0, y0 = corner0.flatten()
            x1, y1 = corner1.flatten()
            d3_coord0 = frame0.get_3dcoord_bilinear(x0, y0)
            d3_coord1 = frame1.get_3dcoord_bilinear(x1, y1)
            if d3_coord0 is not None and d3_coord1 is not None:
                world_points0.append(d3_coord0)
                world_points1.append(d3_coord1)

        # normalize 3d points
        u0 = np.mean(np.array(world_points0), axis=0)
        u1 = np.mean(np.array(world_points1), axis=0)
        world_points0 = world_points0 - u0
        world_points1 = world_points1 - u1

        # following steps are from: https://www.guyuehome.com/36592
        w_mat = np.zeros((3, 3))
        for i, j in zip(world_points0, world_points1):
            w_mat += np.outer(i, j)

        # SVD decomposition
        u, s, vh = np.linalg.svd(w_mat)

        # 0 to 1
        rotMat = u @ vh
        # T_mat = u0 - rotMat @ u1
        T_mat = u1 - rotMat @ u0
        transmat01 = np.eye(4)
        transmat01[:3, :3] = rotMat
        transmat01[:3, 3] = T_mat

        return transmat01

        # 1 to 0
        # rotMat = vh.T @ u.T
        # T_mat = u0 - rotMat @ u1
        # transmat10 = np.eye(4)
        # transmat10[:3, :3] = rotMat
        # transmat10[:3, 3] = T_mat

