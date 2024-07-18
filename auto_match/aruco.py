import cv2
from cv2 import aruco
import numpy as np
from logging import info, debug
import math
from icecream import ic

REVERSE_ARUCO = True
DICT = cv2.aruco.DICT_4X4_250

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    R = math.ceil(math.sqrt((A*A+B*B)))
    return A, B, -C, R

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def validate_aruco_marker(image, markerCorners, markerIds):
    imageCopy = image.copy()

    # Define the dictionary and parameters for ArUco marker detection
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    # Detect ArUco markers
    detectorParams = aruco.DetectorParameters()
    detectorParams.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG

    ret_aruco_corners = []
    ret_aruco_ids = []
    aruco_dict = {}
    for origin_aruco_corners, origin_aruco_id in zip(markerCorners, markerIds):
        aruco_corners = origin_aruco_corners.reshape(-1, 2).astype(np.int32)
        aruco_id = origin_aruco_id[0]
        # the aruco_corners are the four corners of the aruco marker in the image
        # each corner is a point in the image, in the form of (x, y)
        # crop arround the aruco corners, code:
        # 1. get the min and max of x and y of the aruco corners, code:
        min_x, min_y = np.min(aruco_corners, axis=0)
        max_x, max_y = np.max(aruco_corners, axis=0)
        # 2. crop the image using the min and max of x and y, code:
        boundary_x = (max_x - min_x) // 3
        boundary_y = (max_y - min_y) // 3
        min_x = max(0, min_x - boundary_x)
        min_y = max(0, min_y - boundary_y)
        max_x = min(image.shape[1], max_x + boundary_x)
        max_y = min(image.shape[0], max_y + boundary_y)
        crop_drawed = imageCopy[min_y:max_y, min_x:max_x].copy()
        crop_original = image[min_y:max_y, min_x:max_x].copy()

        # detect the aruco in the cropped original image
        crop_afterdraw = crop_original.copy()
        markerCorners, markerIds, _ = aruco.detectMarkers(crop_original, dictionary, parameters=detectorParams)
        # draw the detected markers
        aruco.drawDetectedMarkers(crop_afterdraw, markerCorners, markerIds, (255, 0, 255))
        after_detected_id = markerIds[0][0] if markerIds is not None else None
        crop = np.hstack((crop_drawed, crop_afterdraw, crop_original))

        if aruco_id != after_detected_id:
            print(f"Before val: {aruco_id}, After val: {after_detected_id}")
            # cv2.namedWindow("crop", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("crop", 900, 300)
            # cv2.imshow("crop", crop)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        else:
            # add to the return list, if validated
            ret_aruco_corners.append(origin_aruco_corners)
            ret_aruco_ids.append(origin_aruco_id)

        # add to the dictionary
        aruco_dict[aruco_id] = {
            "id": aruco_id,
            "after_detected_id": after_detected_id,
            "validated": aruco_id == after_detected_id,
            "crop": crop
        }

    return np.array(ret_aruco_corners), np.array(ret_aruco_ids)



class ArucoDetector():
    def __init__(self, img, aruco_dict=DICT):
        """
        img: BGR image from opencv
        indexs_dict: containing aruco id and cooresponding corner number
            i.e. indexs_dict = {
                0: 1,
                1: 4
            }
        aruco_dict: cv provided dict definition
        """
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if REVERSE_ARUCO:
            img_255 = np.full_like(self.img, 255)
            self.img = img_255 - self.img
        self.dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)

        # direct the aruco
        self.corners, self.ids, rejects = cv2.aruco.detectMarkers(self.img, self.dictionary)
        # validate the aruco
        self.corners, self.ids = validate_aruco_marker(self.img, self.corners, self.ids)

        ids = np.squeeze(self.ids)
        corners = np.squeeze(self.corners)

        self.detect_dict = {}
        for id, corners in zip(ids, corners):
            self.detect_dict[id] = corners

        # draw detection result with aruco id
        # if self.corners is not None:
        #     cv2.aruco.drawDetectedMarkers(self.img, self.corners, self.ids)
        #     cv2.namedWindow("aruco", cv2.WINDOW_NORMAL)
        #     cv2.imshow("aruco", self.img)
        #     cv2.waitKey(0)
    
    def get_corners_ids(self):
        corners_lits = []
        for i in range(len(self.corners)):
            corner_refined = cv2.cornerSubPix(self.img, self.corners[i][0], (5, 5), (-1, -1), criteria)
            corners_lits.append(np.expand_dims(corner_refined, 0))
        return np.array(corners_lits)[:,0,:,:], self.ids[:,0]

    def get_centers_ids(self):
        corners_lits = []
        for i in range(len(self.corners)):
            # calculate center point 
            p1, p2, p3, p4 = self.corners[i][0]
            L1 = line(p1, p3)
            L2 = line(p2, p4)
            center = np.array(intersection(L1, L2))

            # refine corner
            corners = np.array([p1,p2,p3,p4, center], dtype=np.float32)
            cv2.cornerSubPix(self.img, corners, (5, 5), (-1, -1), criteria)
            p1, p2, p3, p4, center = corners
            corners_lits.append(center)
        return np.array(corners_lits), self.ids[:,0]

    def get_corners_centers_ids(self):
        corners_lits = []
        for i in range(len(self.corners)):
            # calculate center point 
            p1, p2, p3, p4 = self.corners[i][0]
            L1 = line(p1, p3)
            L2 = line(p2, p4)
            center = np.array(intersection(L1, L2))

            # refine corner
            corners = np.array([p1,p2,p3,p4, center], dtype=np.float32)
            cv2.cornerSubPix(self.img, corners, (5, 5), (-1, -1), criteria)
            # p1, p2, p3, p4, center = corners
            corners_lits.append(corners)
        return np.array(corners_lits), self.ids[:,0]

    def get_aruco_filter_list(self, id_list):
        result_dict = {}
        for target_id in id_list:
            if target_id in self.detect_dict.keys():
                # calculate center point 
                p1, p2, p3, p4 = self.detect_dict[target_id]
                L1 = line(p1, p3)
                L2 = line(p2, p4)
                center = np.array(intersection(L1, L2))

                # refine corner
                corners = np.array([p1,p2,p3,p4, center], dtype=np.float32)
                cv2.cornerSubPix(self.img, corners, (5, 5), (-1, -1), criteria)
                p1, p2, p3, p4, center = corners


                if target_id == 0:
                    coord = p1
                elif target_id == 1:
                    coord = p2
                else:
                    coord = center
                # calculate radius of aruco
                radius = max(L1[3], L2[3])
                result_dict[target_id] = {
                    "coord": coord,
                    "radius": radius
                }
        return result_dict

    def get_aruco_pairs(self, indexs_dict: dict):
        if self.corners is None or self.ids is None:
            return None
        result = []
        for target_id in indexs_dict.keys():
            if target_id in self.detect_dict.keys():
                # calculate center point 
                p1, p2, p3, p4 = self.detect_dict[target_id]
                L1 = line(p1, p3)
                L2 = line(p2, p4)
                center = np.array(intersection(L1, L2))

                # refine corner
                corners = np.array([p1,p2,p3,p4, center], dtype=np.float32)
                cv2.cornerSubPix(self.img, corners, (5, 5), (-1, -1), criteria)
                p1, p2, p3, p4, center = corners

                if target_id == 0:
                    coord = p1
                elif target_id == 1:
                    coord = p2
                else:
                    coord = center
                result.append(coord.astype(np.float64))
            else:
                debug("number of aruco id "+str(target_id)+" has not been detected !")
                return None
            
        return result

                

def detect_aruco(img, index0, index1):
    """
    img: image in bgr
    id: aruco id

    
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if REVERSE_ARUCO:
        img_255 = np.full_like(img, 255)
        img = img_255 - img
    dictionary = cv2.aruco.getPredefinedDictionary(DICT)
    points = []
    index = [index0, index1]
    corners, ids, rejects = cv2.aruco.detectMarkers(img, dictionary)
    print(np.array(corners).shape)
    missing_num = 0

    try:
        corners_index = np.where(ids == index0)
        aruco_corners = corners[corners_index[0][0]]
        points.append(aruco_corners[0][0])
    except IndexError:
        info("number of aruco id "+str(index0)+" has not been detected !")
        missing_num += 1

    try:
        corners_index = np.where(ids == index1)
        aruco_corners = corners[corners_index[0][0]]
        points.append(aruco_corners[0][1])
    except IndexError:
        info("number of aruco id "+str(index1)+" has not been detected !")
        missing_num += 1

    # subpixel
    points = np.array(points, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    points = cv2.cornerSubPix(img, points, (5, 5), (-1, -1), criteria)
    print(list(points).extend([None]*missing_num))
    points = np.squeeze(points)

    return points


def detect_rightimg_corner(img1, img2, index1, index2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    if REVERSE_ARUCO:
        img_255 = np.full_like(img1, 255)
        img1 = img_255 - img1
        img2 = img_255 - img2
    dictionary = cv2.aruco.getPredefinedDictionary(DICT)
    images = [img1, img2]
    index = [index1, index2]
    point = []
    for img in images:
        corners, ids, rejects = cv2.aruco.detectMarkers(img, dictionary)
        try:
            corners_index = np.where(ids == index[0])
            # print(corners[corners_index[0][0]][0][0])
            point.append(corners[corners_index[0][0]][0][0])
        except IndexError:
            print("number of ", index[0], " aruco dose not been detected !")
        try:
            corners_index = np.where(ids == index[1])
            # print(corners[corners_index[0][0]][0][0])
            point.append(corners[corners_index[0][0]][0][1])
        except IndexError:
            print("number of ", index[1], " aruco dose not been detected !")

    # subpixel
    left_img_points = np.array([point[0], point[1]], dtype=np.float32)
    right_img_points = np.array([point[2], point[3]], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    left_img_points = cv2.cornerSubPix(img1, left_img_points, (5, 5), (-1, -1), criteria)
    left_img_points = np.squeeze(left_img_points)
    right_img_points = cv2.cornerSubPix(img2, right_img_points, (5, 5), (-1, -1), criteria)
    right_img_points = np.squeeze(right_img_points)

    return left_img_points, right_img_points



if __name__ == "__main__":
    img1 = cv2.imread(r"C:\Users\Andy\DCIM\A_21317.jpg")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    img2 = cv2.imread(r"C:\Users\Andy\DCIM\D_21317.jpg")
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    point = detect_rightimg_corner(img1, img2, 0, 1)
    print(point)