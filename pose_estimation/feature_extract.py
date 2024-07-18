import cv2
import numpy as np
import time

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
    image_gray = cv2.equalizeHist(image_gray)
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
