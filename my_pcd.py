import open3d as o3d
import numpy as np
import copy
import cv2
from pathlib import Path
from icecream import ic
from typing_extensions import Self
from auto_match.aruco import ArucoDetector

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
        result = o3d.pipelines.registration.registration_icp(
            source_down, 
            target_down, 
            radius, 
            current_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
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
        matches01 = my_match(kps0, kps1, features0, features1, ratio=0.7)

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

        # DISTORTION:
        # K1:-0.0564663522    K2:0.1112944335    K3:-0.0645765141    
        # P1:-0.0001750350    P2:0.0002957206    
        dist = np.array([-0.0564663522, 0.1112944335, -0.0001750350, 0.0002957206, -0.0645765141])

        success, rvec, tvec, inliers = cv2.solvePnPRansac(world_points, frame1_points, cm, dist)
        # Use right image to refine
        if success and inliers is not None:
            rvec, tvec = cv2.solvePnPRefineLM(world_points[np.squeeze(inliers)], frame1_points[np.squeeze(inliers)], cm, dist, rvec, tvec)

        rotMat, _ = cv2.Rodrigues(np.array(rvec))
        T_mat_opencv = np.eye(4)
        T_mat_opencv[:3,:3] = rotMat
        T_mat_opencv[:3, 3:] = tvec
        # print("T_mat_opencv: \n", T_mat_opencv)
        return T_mat_opencv

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

    def estimate_RT_aruco_optimize(self, pcd2: Self):
        """
        Estimate the transformation matrix between two frames using aruco markers
        R, T optimization using scipy minimize
        """
        frame0, frame1 = self, pcd2
        image0, image1 = frame0.image, frame1.image

        ad0 = ArucoDetector(image0)
        ad1 = ArucoDetector(image1)
        corners0, ids0 = ad0.get_corners_centers_ids()
        corners1, ids1 = ad1.get_corners_centers_ids()

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

        # scipy optimize
        from scipy.optimize import minimize
        from scipy.spatial.transform import Rotation as R
        def loss(x):
            rot = R.from_euler('xyz', x[:3]).as_matrix()
            T = x[3:]
            error = 0
            for i, j in zip(world_points0, world_points1):
                error += np.linalg.norm(rot @ i + T - j)
            return error

        # initial guesses
        x0 = np.zeros(6)
        # x0 = np.random.rand(6)
        # x0 = np.array([1, 1, 1, 0, 0, 0])
        res = minimize(loss, x0, method='Nelder-Mead')
        # print(res)
        # print final error
        print("Final error: ", loss(res.x))
        rot = R.from_euler('xyz', res.x[:3]).as_matrix()
        T = res.x[3:] + u1 - rot @ u0
        transmat01 = np.eye(4)
        transmat01[:3, :3] = rot
        transmat01[:3, 3] = T
        return transmat01

    def get_aruco_coords(self):
        """
        Get the 3d coord of every aruco markers' center in the frame
        """
        ad = ArucoDetector(self.image)
        points_2d, ids = ad.get_corners_centers_ids()
        points_2d = points_2d[:,-1,:]

        world_points = []
        save_indexes = np.ones_like(ids, dtype=bool)

        for i in range(len(points_2d)):
            p_3d = self.get_3dcoord_bilinear(points_2d[i,0], points_2d[i,1])
            if p_3d is not None:
                world_points.append(p_3d)
            else:
                save_indexes[i] = False

        ids = ids[save_indexes]
        points_2d = points_2d[save_indexes]
        world_points = np.array(world_points)
        self.points_2d = points_2d

        return world_points, ids

    # use CCT decoder 
    def estimate_RT_CCT_optimize(self, pcd2: Self):
        """
        Estimate the transformation matrix between two frames using CCT markers
        R, T optimization using scipy minimize
        """
        from CCTDecoder.cct_decode import CCT_extract
        frame0 = self
        frame1 = pcd2
        img0 = frame0.image
        img1 = frame1.image

        # extract CCT id and coords
        N = 8
        cct0, _ = CCT_extract(img0, 8)
        cct1, _ = CCT_extract(img1, 8)

        # get common cct ids
        common_ids = set(cct0.keys()).intersection(set(cct1.keys()))
        # ic(len(common_ids), common_ids)
        
        if __name__ == "__main__":
            # show matches in image
            concant_img = np.concatenate((img0, img1), axis=0)
            # draw matches on the concant image, using line and circle
            for id in common_ids:
                x0, y0 = cct0[id]
                x1, y1 = cct1[id]
                cv2.circle(concant_img, np.int16([x0, y0]), 5, (0, 0, 255), -1)
                cv2.circle(concant_img, np.int16([x1, y1+frame0.h]), 5, (0, 0, 255), -1)
                cv2.line(concant_img, np.int16([x0, y0]), np.int16([x1, y1+frame0.h]), (0, 255, 0), 1)
                # cv2.line(concant_img, np.int16([x0, y0]), np.int16([x1+frame0.w, y1]), (0, 255, 0), 1)
            # show using pil
            from PIL import Image
            Image.fromarray(concant_img).show()

        # get 3d points from cct ids
        world_points0 = []
        world_points1 = []
        for id in common_ids:
            d3_coord0 = frame0.get_3dcoord_bilinear(*cct0[id])
            d3_coord1 = frame1.get_3dcoord_bilinear(*cct1[id])
            if d3_coord0 is not None and d3_coord1 is not None:
                world_points0.append(d3_coord0)
                world_points1.append(d3_coord1)
            else:
                if d3_coord0 is None:
                    print(f"Failed to get 3d coord for id {id} in {frame0.folder_path}")
                if d3_coord1 is None:
                    print(f"Failed to get 3d coord for id {id} in {frame1.folder_path}")
                                 
        # normalize 3d points
        u0 = np.mean(np.array(world_points0), axis=0)
        u1 = np.mean(np.array(world_points1), axis=0)
        world_points0 = world_points0 - u0
        world_points1 = world_points1 - u1

        # scipy optimize
        from scipy.optimize import minimize
        from scipy.spatial.transform import Rotation as R
        def loss(x):
            rot = R.from_euler('xyz', x[:3]).as_matrix()
            T = x[3:]
            error = 0
            for i, j in zip(world_points0, world_points1):
                error += np.linalg.norm(rot @ i + T - j)
            return error

        # initial guesses
        # r =  R.from_matrix(rotMat)
        # rot_eular = r.as_euler("xyz",degrees=False)
        # x0 = np.concatenate((rot_eular, T_mat))
        x0 = np.zeros(6)
        # x0 = np.random.rand(6)
        # x0 = np.array([1, 1, 1, 0, 0, 0])
        res = minimize(loss, x0, method='Nelder-Mead')
        # print(res)
        # print final error
        print("Final error: ", loss(res.x))
        rot = R.from_euler('xyz', res.x[:3]).as_matrix()
        T = res.x[3:] + u1 - rot @ u0
        transmat01 = np.eye(4)
        transmat01[:3, :3] = rot
        transmat01[:3, 3] = T
        return transmat01

if __name__ == "__main__":
    pcd1 = MyPCD(r"C:\workspace\data\2.85m\2.85m_5\img\4")
    pcd0 = MyPCD(r"C:\workspace\data\2.85m\2.85m_5\img\5")

    rt = pcd0.estimate_RT_aruco_optimize(pcd1)

    print(rt)

    # visual
    draw_registration_result(pcd0.pcd, pcd1.pcd, rt)

    # Concatenate the point clouds
    pcd_combined = o3d.geometry.PointCloud()
    pcd_combined += pcd0.pcd.transform(rt)
    pcd_combined += pcd1.pcd
    o3d.io.write_point_cloud("combined_pointcloud.ply", pcd_combined)
