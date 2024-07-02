import numpy as np
import open3d as o3d
from icecream import ic
from my_pcd import MyPCD, colored_icp_registration
from typing import List
from pathlib import Path
from utils import timeit

CM = np.array([[1746.8647460938, 0.0000000000, 1018.6414794922],
                [0.0000000000, 1745.8187255859, 793.8147583008],
                [0.0000000000, 0.0000000000, 1.0000000000]])

@timeit
def _old_method():
    # Load the point clouds
    frame0 = MyPCD(r"Z:\andyls\D3_image\frame0")
    frame1 = MyPCD(r"Z:\andyls\D3_image\frame1")
    frame2 = MyPCD(r"Z:\andyls\D3_image\frame2")

    rt_01 = frame0.estimat_RT_pnp(frame1, CM)
    rt_12 = frame1.estimat_RT_pnp(frame2, CM)

    ic(rt_01)
    ic(rt_12)

    rt_01_refined = frame0.refine_RT_icp(frame1, rt_01)
    rt_12_refined = frame1.refine_RT_icp(frame2, rt_12)

    ic(rt_01_refined)
    ic(rt_12_refined)

    # Concatenate the two point clouds
    pcd_combined = o3d.geometry.PointCloud()
    pcd_combined += frame0.pcd.transform(rt_01_refined).transform(rt_12_refined)
    pcd_combined += frame1.pcd.transform(rt_12_refined)
    pcd_combined += frame2.pcd

    # Save the combined point cloud
    o3d.io.write_point_cloud("combined_pointcloud.ply", pcd_combined)


@timeit
def _new_method():
    # read in frames
    frame0 = MyPCD(r"Z:\andyls\D3_image\frame0")
    frame1 = MyPCD(r"Z:\andyls\D3_image\frame1")
    frame2 = MyPCD(r"Z:\andyls\D3_image\frame2")

    rt_01 = frame0.estimate_RT_aruco_icp(frame1)
    rt_12 = frame1.estimate_RT_aruco_icp(frame2)

    rt_01_refined = colored_icp_registration(frame0.pcd, frame1.pcd, 0.001, rt_01)
    rt_12_refined = colored_icp_registration(frame1.pcd, frame2.pcd, 0.001, rt_12)


    # Concatenate the two point clouds
    pcd_combined = o3d.geometry.PointCloud()
    pcd_combined += frame0.pcd.transform(rt_01_refined).transform(rt_12_refined)
    pcd_combined += frame1.pcd.transform(rt_12_refined)
    pcd_combined += frame2.pcd

    # Save the combined point cloud
    o3d.io.write_point_cloud("combined_pointcloud.ply", pcd_combined)

# _old_method()
# _new_method()


# function that inputs a list of MyPCD objects and outputs:
# 1. the combined point cloud
# 2. the refined transformation matrices from each frame to the first frame
def combine_frames(frames: List[MyPCD]):
    rt_dict = {}

    # This is the 0-0 transformation matrix
    rt_dict[frames[0].folder_path.name] = np.eye(4)

    for i in range(1, len(frames)):
        # i-th frame to i-1-th frame transformation matrix
        source = frames[i]
        target = frames[i-1]

        rt = source.estimate_RT_aruco_icp(target)
        # rfine the transformation matrix
        rt_refined = colored_icp_registration(source.pcd, target.pcd, 0.001, rt)

        # i-th frame to 0-th frame transformation matrix
        rt_dict[frames[i].folder_path.name] = rt_refined @ rt_dict[frames[i-1].folder_path.name]


    # Concatenate the point clouds
    pcd_combined = o3d.geometry.PointCloud()
    for i in range(len(frames)):
        pcd_combined += frames[i].pcd.transform(rt_dict[frames[i].folder_path.name])

    return pcd_combined, rt_dict


if __name__ == '__main__':
    combine_folder = Path(r"C:\workspace\data\D3_image")

    # glob all the folders
    frame_folders = [f for f in combine_folder.iterdir() if f.is_dir()]
    ic(frame_folders)
    frames = [MyPCD(f) for f in frame_folders]

    pcd_combined, rt_list = combine_frames(frames)
    o3d.io.write_point_cloud(str(combine_folder / "combined_pointcloud.ply"), pcd_combined)
    ic(rt_list)

    import pickle
    # save the rt_list to the folder
    with open(combine_folder / "rt_list.pkl", "wb") as f:
        pickle.dump(rt_list, f)
