from collections import defaultdict
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os


def read_and_resize(px_y, imgfile):
    img = cv2.imread(imgfile)
    scale = px_y / img.shape[0]
    return cv2.resize(img, (0, 0), fx=scale, fy=scale)


def show(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def bgr_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def linesP(canny, img):

    img = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    y_size = canny.shape[1]
    x_size = canny.shape[0]
    canny[:y_size // 6, :] = 0
    canny[:, :x_size // 6] = 0
    canny[5*y_size // 6:, :] = 0
    canny[:, 5*x_size // 6:] = 0

    lines = cv2.HoughLinesP(canny, 1, np.pi/180, 10, None, 100, 20)

    if lines is not None:
        lines = np.squeeze(lines)
        for i in range(0, len(lines)):
            x1, y1, x2, y2 = lines[i]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3, cv2.LINE_AA)
            cv2.circle(img, (x1, y1), 2, (255, 0, 255), 3)
            cv2.circle(img, (x2, y2), 2, (255, 0, 255), 3)

    return img


def locate_grid(canny, img):

    grid = img
    grid[:, :] = 0
    img = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

    y_size = canny.shape[1]
    x_size = canny.shape[0]
    canny[:y_size // 6, :] = 0
    canny[:, :x_size // 6] = 0
    canny[5*y_size // 6:, :] = 0
    canny[:, 5*x_size // 6:] = 0

    lines = cv2.HoughLines(canny, 1, np.pi/180, 90)

    if lines is not None:
        lines = np.squeeze(lines)
        grid_lines = [lines[0]]
        for i in range(0, len(lines)):
            is_grid_line = True
            for j in range(0, len(grid_lines)):
                d_rho = abs(lines[i][0] - grid_lines[j][0])
                d_theta = abs(lines[i][1] - grid_lines[j][1])
                if d_rho >= 0 and d_rho < 100 and d_theta < 10*np.pi/180:
                    is_grid_line = False
            if (is_grid_line):
                grid_lines.append(lines[i])

        grid_lines = grid_lines[:4]

        # https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv
        lines, lines_0, lines_1 = segment_by_angle_kmeans(grid_lines)

        intersections = list(segmented_intersections(lines))

        for line in lines_0:
            pt1, pt2 = construct_line(line)
            cv2.line(img, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(grid, pt1, pt2, (255, 255, 255), 2, cv2.LINE_AA)

        for line in lines_1:
            pt1, pt2 = construct_line(line)
            cv2.line(img, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(grid, pt1, pt2, (255, 255, 255), 2, cv2.LINE_AA)

        for pt in intersections:
            cv2.circle(img, (pt[0], pt[1]), 10, (255, 255, 255), 10)
        grid = cv2.dilate(grid, None)

        intersections.sort(key=lambda x: x[1])

    return img, intersections


def construct_line(line):
    rho = line[0]
    theta = line[1]
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    return pt1, pt2


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))
    intersections = np.array(intersections)
    intersections = np.squeeze(intersections)
    return intersections


def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle 
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, _ = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in enumerate(lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    lines0 = np.array(segmented[0])
    lines0 = np.squeeze(lines0)
    lines1 = np.array(segmented[1])
    lines1 = np.squeeze(lines1)
    return segmented, lines0, lines1


def lines(canny, img):

    grid = img
    grid[:, :] = 0
    img = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

    y_size = canny.shape[1]
    x_size = canny.shape[0]
    canny[:y_size // 6, :] = 0
    canny[:, :x_size // 6] = 0
    canny[5*y_size // 6:, :] = 0
    canny[:, 5*x_size // 6:] = 0

    lines = cv2.HoughLines(canny, 1, np.pi/180, 90)

    if lines is not None:
        lines = np.squeeze(lines)
        grid_lines = [lines[0]]
        for i in range(0, len(lines)):
            is_grid_line = True
            for j in range(0, len(grid_lines)):
                d_rho = abs(lines[i][0] - grid_lines[j][0])
                d_theta = abs(lines[i][1] - grid_lines[j][1])
                if d_rho >= 0 and d_rho < 100 and d_theta < 10*np.pi/180:
                    is_grid_line = False
            if (is_grid_line):
                grid_lines.append(lines[i])

        grid_lines = grid_lines[0:4]
        for line in grid_lines:
            rho = line[0]
            theta = line[1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(img, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(grid, pt1, pt2, (255, 255, 255), 2, cv2.LINE_AA)
        grid = cv2.dilate(grid, None)

    return img, grid


def corners(grid):

    corners, grid = sort_corners(grid)

    grid = rotate(grid, corners)

    '''
    kernel = 5
    element = cv2.getStructuringElement(
    cv2.MORPH_RECT, (2*kernel + 1, 2*kernel + 1), (kernel, kernel))

    grid = cv2.dilate(grid, element)
    '''

    _, _ = sort_corners(grid)

    grid = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)

    for i in range(0, len(corners)):
        cv2.circle(grid, (corners[i][0], corners[i][1]), 10, (0, 255, 0), 10)

    return corners, grid


def sort_corners(grid, intersections):

    indices = [None, None, None, None]
    selected = [False, False, False, False]

    for corner in range(4):
        sort_corners = [None, None, None, None]
        for i in range(len(intersections)):
            tl = math.sqrt(
                pow(intersections[i][0], 2) + pow(intersections[i][1], 2))
            tr = math.sqrt(
                pow(intersections[i][0] - grid.shape[1], 2) + pow(intersections[i][1], 2))
            bl = math.sqrt(pow(intersections[i][0], 2) +
                           pow(intersections[i][1] - grid.shape[0], 2))
            br = math.sqrt(pow(intersections[i][0] - grid.shape[1], 2) +
                           pow(intersections[i][1] - grid.shape[0], 2))

            distances = [tl, tr, bl, br]

            sort_corners[i] = distances[corner]

        min = 10000
        for i in range(len(sort_corners)):
            if not selected[i] and sort_corners[i] < min:
                min = sort_corners[i]
                indices[corner] = i
        selected[indices[corner]] = True

    intersections = [intersections[i] for i in indices]

    return intersections


def rotate(grid, corners, img=None):
    # rotate points
    # https://stackoverflow.com/questions/7953316/rotate-a-point-around-a-point-with-opencv
    co = abs(corners[2][0] - corners[3][0])
    hyp = math.sqrt(pow(corners[2][0] - corners[3][0], 2) +
                    pow(corners[2][1] - corners[3][1], 2))
    angle = math.acos(co / hyp)

    h, w = grid.shape[:2]

    rotated_corners = [corners[:][0] *
                       math.cos(angle) + corners[:][1]*math.sin(angle)]

    center = (w/2, h/2)

    coord_t = []

    for corner in corners:
        coord = (corner[0] - center[0], corner[1] - center[1])
        x_t = coord[0] * math.cos(angle) + coord[1] * \
            math.sin(angle) + center[0]
        y_t = - coord[0] * math.sin(angle) + \
            coord[1] * math.cos(angle) + center[1]
        coord_t.append((round(x_t), round(y_t)))

    rotate_matrix = cv2.getRotationMatrix2D(
        center=center, angle=math.degrees(angle), scale=1)

    # cv2.circle(grid, (int(corners[0][0]), int(corners[0][1])), 10, (255, 255, 0), 10)

    grid = cv2.warpAffine(src=grid, M=rotate_matrix, dsize=(w, h))

    if img is not None:
        img = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(w, h))

    for corner in coord_t:
        cv2.circle(grid, corner, 10, (0, 255, 255), 10)

    return grid, coord_t, img


def zoning(corners, img, print=False):

    corners = sort_corners(img, corners)

    zone1 = img[:corners[0][1], :corners[0][0]]
    zone2 = img[:corners[0][1], corners[0][0]:corners[1][0]]
    zone3 = img[:corners[0][1], corners[1][0]:]
    zone4 = img[corners[0][1]:corners[2][1], :corners[2][0]]
    zone5 = img[corners[0][1]:corners[2]
                [1], corners[2][0]:corners[3][0]]
    zone6 = img[corners[1][1]:corners[3][1], corners[1][0]:]
    zone7 = img[corners[2][1]:, :corners[2][0]]
    zone8 = img[corners[2][1]:, corners[2][0]:corners[3][0]]
    zone9 = img[corners[2][1]:, corners[3][0]:]

    zones = [zone1, zone2, zone3, zone4, zone5, zone6, zone7, zone8, zone9]

    if print:
        i = 1
        plt.figure()
        for zone in zones:
            plt.subplot(3, 3, i)
            show(zone)
            i += 1

    return zones


def export(img, prefix):

    paths = []
    i = 1
    os.makedirs("img/generated/" + prefix, exist_ok=True)
    plt.figure()
    for im in img:
        im = bgr_gray(im)
        kernel = 7
        element = cv2.getStructuringElement(
            cv2.MORPH_RECT, (2*kernel + 1, 2*kernel + 1), (kernel, kernel))

        im = cv2.erode(im, element)
        filepath = "img/generated/" + prefix + \
            "/" + prefix + "_" + str(i) + ".png"
        paths.append(filepath)
        cv2.imwrite(filepath, im)
        plt.subplot(3, 3, i)
        show(im)
        i += 1

    return paths