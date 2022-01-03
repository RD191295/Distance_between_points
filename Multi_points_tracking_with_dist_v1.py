from __future__ import print_function
import sys
from typing import Tuple

import cv2
import math
from random import randint
from scipy.spatial import distance as dist

trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.legacy.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.legacy.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.legacy.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.legacy.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker


cap = cv2.VideoCapture("Input_Video/DemoVideo.mp4")

success, frame = cap.read()

# SIZE
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

# VIDEO WRITER
result = cv2.VideoWriter('output_distance.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

if not success:
    print('Failed to read video')
    sys.exit(1)

## Select boxes
bboxes = []
colors = []
mid_points = []
dist_lists = []

while True:
    bbox = cv2.selectROI('MultiTracker', frame)
    bboxes.append(bbox)
    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    print("Press q to quit selecting boxes and start tracking")
    print("Press any other key to select next object")
    k = cv2.waitKey(0) & 0xFF
    print(k)
    if k == 113:  # q is pressed
        break

trackerType = "CSRT"
createTrackerByName(trackerType)

# Create MultiTracker object
multiTracker = cv2.MultiTracker_create()

# Initialize MultiTracker
for bbox in bboxes:
    multiTracker.add(createTrackerByName(trackerType), frame, bbox)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    success, boxes = multiTracker.update(frame)
    total_points = len(boxes)
    if success:
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            # cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
            mid_point = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
            cv2.circle(frame, mid_point, 5, (0, 0, 255), -1)
            cv2.putText(frame, "p" + str(i), (mid_point[0] + 5, mid_point[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 0, 255), 2)
            mid_points.append(mid_point)
            if len(mid_points) == total_points:
                number_of_line = int(total_points / 2)
                for i in range(number_of_line+1):
                    cv2.line(frame, mid_points[i], mid_points[i + 1], (0, 255, 0))
                    Distance = dist.euclidean(mid_points[i], mid_points[i + 1])
                    dist_lists.append(Distance)

                initial_x = 20
                initial_y = 40
                for i in range(len(dist_lists)):
                    cv2.putText(frame, "distance between p" + str(i) + "and p" + str(i + 1) + " is: "+str(round(dist_lists[i], 3)), (initial_x, initial_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                    initial_y = initial_y + 20
                mid_points = []
                dist_lists = []


    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    result.write(frame)
    cv2.imshow('MultiTracker', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        break

cap.release()
result.release()

cv2.destroyAllWindows()
