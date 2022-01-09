import numpy as np
import tensorflow as tf
import keras
import cv2
import logging

# filters the frame and draws out edges of lane lines
def lane_detection(frame):
    # convert from BGR to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # isolate the lane color (blue in this case)
    lowest_blue = np.array([65, 30, 30])
    highest_blue = np.array([145, 255, 255])
    masked_frame = cv2.inRange(hsv_frame, lowest_blue, highest_blue)

    # find edges of lane lines
    edges = cv2.Canny(masked_frame, 200, 400)
    return edges

# ignores any noise from the top half of the frame and only focuses on the bottom half
def crop_frame(edges):
    height, width = edges.shape

    # we only want to focus on the bottom half of our frame because that's where the lanes will be
    polygon = np.array([[
        (0, height * 1 / 2),
        (width, height * 1 / 2),
        (width, 0),
        (0, 0),
    ]], np.int32)
    cropped_edges = cv2.fillPoly(edges, polygon, 0)

    return cropped_edges

# convert edges to lane lines using houghLinesP and merge all the detected lines into
# the two main lane lines that we care about most
def line_detection(cropped_edges):
    rho = 1
    angle = np.pi/180
    min_threshold = 10
    lines = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold,
                            np.array([]), minLineLength=8, maxLineGap=4)

    #combine all the line segments we detected and average them into the two main lane lines
    return lines

def compute_average_slopes(frame, lines):
    lane_lines = []
    if lines is None:
        logging.info('No line_segment segments detected')
        return lane_lines

    height, width = frame.shape
    left_fit = []
    right_fit = []

    # make sure we are detecting the correct line segments
    # left lane line segment should be on left 2/3 of the screen
    # right lane line segment should be on left 2/3 of the screen
    boundary = 1/3
    left_region_boundary = width * (1 - boundary)
    right_region_boundary = width * boundary

    for line in lines:
        for x1, y1, x2, y2 in line:

            # skip vertical lines
            if x1 == x2:
                logging.info('skipping vertical line segment (slope=inf): %s' % line)
                continue

            # use np.polyfit to find slope and intercept of each line
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]

            # check if each line is within its boundaries
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    #find the average of the slopes and intersects of the lines on the left
    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))
    #find the average of the slopes and intersects of the lines on the right
    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    logging.debug('lane lines: %s' % lane_lines)

    return lane_lines

# helper function for compute_average_slopes()
# converts the resulting averagen lines from np.polyfit() back into the format of [x1,y1,x2,y2]
# for use in future functions
def make_points(frame, line):
    height, width = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame downwards

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]

def display_lines(frame, lines, line_color=(0, 255, 0), line_width=2):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image



'''
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    frame = lane_detection(frame)
    frame = crop_frame(frame)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
'''

frame = cv2.imread("test.JPG")
lanes_frame = lane_detection(frame)
cropped_frame = crop_frame(lanes_frame)
lines = line_detection(cropped_frame)
lane_lines = compute_average_slopes(cropped_frame, lines)
lane_lines_image = display_lines(frame, lane_lines)
cv2.imwrite("target.jpg", lane_lines_image)




