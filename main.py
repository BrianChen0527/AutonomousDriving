import numpy as np
import tensorflow as tf
import keras
import cv2
import logging
import math

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

# Overlay the lane lines with the original frame
def display_lines(frame, lines, line_color=(0, 255, 0), line_width=14):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(line_image, 1, frame, 1, 1)
    return line_image

def find_steering_angle(frame, lines):
    height, width, _ = frame.shape
    x_offset = 0
    y_offset = 0

    if(len(lines) == 2):
        _, _, left_lane_x2, _ = lines[0][0]
        _, _, right_lane_x2, _ = lines[1][0]
        x_offset = int((left_lane_x2 + right_lane_x2)/2 - width/2)
        y_offset = int(height/2)
    else:
        x1, _, x2, _ = lines[0][0]
        x_offset = int(x2 - x1)
        y_offset = int(height / 2)

    #compute the steering angle and then convert it from radians to degrees
    angle_to_mid = int(math.atan(x_offset/y_offset)*180./np.pi)
    steering_angle = 90 + angle_to_mid
    return steering_angle

# figure out the heading line from steering angle
def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=14):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    # Steering direction of angles:
    # 0-89 degree: turn left
    # 90 degree: going straight
    # 91-180 degree: turn right
    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image


def stabilize_steering_angle(curr_steering_angle,
                            new_steering_angle,
                            num_of_lane_lines,
                            max_angle_deviation_two_lines=5,
                            max_angle_deviation_one_lane=15):
    '''
    Using last steering angle to stabilize the steering angle
    if new angle is too different from current angle,
    only turn by max_angle_deviation degrees
    '''
    if num_of_lane_lines == 2:
        # if both lane lines detected, then we can deviate more
        max_angle_deviation = max_angle_deviation_two_lines
    else:
        # if only one lane detected, we need to turn quicker
        max_angle_deviation = max_angle_deviation_one_lane

    angle_deviation = new_steering_angle - curr_steering_angle
    if abs(angle_deviation) > max_angle_deviation:
        stabilized_steering_angle = int(curr_steering_angle
                                        + max_angle_deviation * angle_deviation / abs(angle_deviation))
    else:
        stabilized_steering_angle = new_steering_angle
    return stabilized_steering_angle


# live video capture display + steering
def drive():
    cap = cv2.VideoCapture(0)

    curr_steering_angle = 90
    while(True):
        ret, frame = cap.read()

        lanes_frame = lane_detection(frame)
        cropped_frame = crop_frame(lanes_frame)
        lines = line_detection(cropped_frame)
        lane_lines = compute_average_slopes(cropped_frame, lines)
        lane_lines_frame = display_lines(frame, lane_lines)
        new_steering_angle = find_steering_angle(lane_lines_frame, lane_lines)
        steering_frame = display_heading_line(lane_lines_frame, new_steering_angle)

        stabilized_steering_angle = stabilize_steering_angle(curr_steering_angle,
                                                            new_steering_angle,
                                                            len(lines))
        curr_steering_angle = stabilized_steering_angle
        print("[+] current steering angle: " + str(curr_steering_angle))

        cv2.imshow('frame', steering_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

drive()



'''
# test portion that runs on individual images
frame = cv2.imread("test.JPG")
lanes_frame = lane_detection(frame)
cropped_frame = crop_frame(lanes_frame)
lines = line_detection(cropped_frame)
lane_lines = compute_average_slopes(cropped_frame, lines)
print(lane_lines)
lane_lines_image = display_lines(frame, lane_lines)
steering_angle = find_steering_angle(lane_lines_image, lane_lines)
steering_image = display_heading_line(lane_lines_image, steering_angle)

stabilized_steering_angle = stabilize_steering_angle()
cv2.imwrite("target.jpg", steering_image)
'''



