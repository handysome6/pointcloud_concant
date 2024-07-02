import cv2
import numpy as np
import time
import copy
from matplotlib import pyplot as plt
from python_orb_slam3 import ORBExtractor

def opencv_trible_match(kpsA, kpsB, kpsC, matches_AB, matches_AC):
    matches_array_AB = np.array([m.queryIdx for m in matches_AB])
    matches_array_AC = np.array([m.queryIdx for m in matches_AC])
    common_matched_A = np.intersect1d(matches_array_AB, matches_array_AC)

    frame1_left_points = []
    frame1_right_points = []
    frame2_left_points = []
    new_matches_AB = []
    for m in matches_AB:
        if m.queryIdx in common_matched_A:
            new_matches_AB.append(m)
            frame1_left_points.append([kpsA[m.queryIdx].pt[0], kpsA[m.queryIdx].pt[1]])
            frame1_right_points.append([kpsB[m.trainIdx].pt[0], kpsB[m.trainIdx].pt[1]])

    for m in matches_AC:
        if m.queryIdx in common_matched_A:
            frame2_left_points.append([kpsC[m.trainIdx].pt[0], kpsC[m.trainIdx].pt[1]])
    
    frame1_left_points = np.array(frame1_left_points)
    frame1_right_points = np.array(frame1_right_points)
    frame2_left_points = np.array(frame2_left_points)

    return frame1_left_points, frame1_right_points, frame2_left_points

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

# def draw_points_reprojection(pixel_left, pixel_right, left_points_in_raw, right_points_in_raw, undistorted_left_image, undistorted_right_image):
#     left_img = copy.copy(undistorted_left_image)
#     right_img = copy.copy(undistorted_right_image)
#     point_size = 1
#     point_color = (0, 0, 255) # BGR
#     thickness = 4

#     for i,point in enumerate(pixel_left):
#         cv2.circle(left_img, (int(point[0]), int(point[1])), point_size, point_color, thickness)

#     for i,point in enumerate(left_points_in_raw):        
#         cv2.circle(left_img, (int(point[0]), int(point[1])), point_size, (0, 255, 0), thickness)
    
#     for i,point in enumerate(pixel_right):
#         cv2.circle(right_img, (int(point[0]), int(point[1])), point_size, point_color, thickness)

#     for i,point in enumerate(right_points_in_raw):        
#         cv2.circle(right_img, (int(point[0]), int(point[1])), point_size, (0, 255, 0), thickness)

#     show_img = cv2.hconcat([left_img, right_img])
#     return show_img

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

def det_chessboard_corners(left_img):
    img1 = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    ret1, corner1 = cv2.findChessboardCorners(img1, (11,8))
    ret1, corner1 = cv2.find4QuadCornerSubpix(img1, corner1, (7,7))

    cv2.drawChessboardCorners(img1, (11,8), corner1, ret1)
    cv2.namedWindow("left_corner", 0)
    cv2.resizeWindow("left_corner", 1600, 800)  # 设置窗口大小    
    cv2.imshow('left_corner', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return corner1.squeeze()

def orb_slam3_matcher(path1, path2):
    source = cv2.imread(path1)
    target = cv2.imread(path2)

    orb_extractor = ORBExtractor()

    # Extract features from source image
    source_keypoints, source_descriptors = orb_extractor.detectAndCompute(source)
    target_keypoints, target_descriptors = orb_extractor.detectAndCompute(target)

    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(source_descriptors, target_descriptors)

    # matches = matchKeyPointsKNN(source_descriptors, target_descriptors, ratio=0.5, method='orb')

    # Draw matches
    source_image = cv2.drawKeypoints(source, source_keypoints, None)
    target_image = cv2.drawKeypoints(target, target_keypoints, None)
    matches_image = cv2.drawMatches(source_image, source_keypoints, target_image, target_keypoints, matches, None)
    cv2.namedWindow("Match", 0)
    cv2.resizeWindow("Match", 1400, 600)  # 设置窗口大小
    cv2.imshow("Match", matches_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 选择特征提取器函数
def detectAndDescribe(image, method=None):
    if method == 'sift':
        descriptor = cv2.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()  # OpenCV4以上不可用
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()
    elif method == 'akaze':
        descriptor = cv2.AKAZE_create()
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (kps, features) = descriptor.detectAndCompute(image_gray, None)
    return kps, features


# 创建匹配器
def createMatcher(method, crossCheck):
    """
    不同的方法创建不同的匹配器参数，参数释义
        BFMatcher：暴力匹配器
        NORM_L2-欧式距离
        NORM_HAMMING-汉明距离
        crossCheck-若为True，即两张图像中的特征点必须互相都是唯一选择
    """
    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk' or method == 'akaze':
        # 创建BF匹配器
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        # 创建Flann匹配器
        bf = cv2.FlannBasedMatcher(index_params, search_params)
    return bf


# 暴力检测函数
def matchKeyPointsBF(featuresA, featuresB, method):
    start_time = time.time()
    bf = createMatcher(method, crossCheck=True)
    best_matches = bf.match(featuresA, featuresB)
    rawMatches = sorted(best_matches, key=lambda x: x.distance)
    print("Raw matches (Brute force):", len(rawMatches))
    end_time = time.time()
    print("暴力检测共耗时" + str(end_time - start_time))
    return rawMatches


# 使用knn检测函数
def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
    start_time = time.time()
    bf = createMatcher(method, crossCheck=False)
    # rawMatches = bf.knnMatch(featuresA, featuresB, k=2)
    # 上面这行在用Flann时会报错
    rawMatches = bf.knnMatch(np.asarray(featuresA, np.float32), np.asarray(featuresB, np.float32), k=2)
    matches = []
    for m, n in rawMatches:
        if m.distance < n.distance * ratio:
            matches.append(m)
    print(f"knn匹配的特征点数量:{len(matches)}")
    end_time = time.time()
    print("KNN检测共耗时" + str(end_time - start_time))
    return matches

def my_match(kp1, kp2, featuresA, featuresB, ratio=0.5):
    # FLANN参数
    MIN_MATCH_COUNT = 10
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(featuresA, featuresB, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, method=cv2.FM_RANSAC,ransacReprojThreshold=0.9, confidence=0.99)

        matchesMask = mask.ravel().tolist()
        # print(len(np.nonzero(matchesMask)[0]))

    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    selected_good = [m for m, cond in zip(good, matchesMask) if cond>0]
    return selected_good
