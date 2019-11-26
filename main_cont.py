import cv2
import numpy as np
import collections as col
#global variables
MaxTrackCount = 50
Buffer = 32
MinCountourArea = 5000 #Adjust ths value according to your usage
counter = 0
# video = cv2.VideoCapture("people-walking.mp4")
# video = cv2.VideoCapture("4p-c2.avi")
# video = cv2.VideoCapture("crosswalk.avi")
video = cv2.VideoCapture("TownCentreXVID.avi")
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=80, history=50, detectShadows=0)

pts = col.deque(maxlen=Buffer)
traj = col.deque(maxlen=MaxTrackCount)
while True:
    _, frame = video.read()

    fgmask = fgbg.apply(frame)
    cv2.imshow('fgmask', fgmask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilation = cv2.dilate(fgmask, kernel, iterations=5)
    # opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    ckernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # eroded = cv2.erode(dilation, ckernel, iterations=5)
    cv2.imshow('dilation', dilation)
    _, cnts, _ = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    QttyOfContours = 0

    term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    # check all found countours
    for c in cnts:
        # if a contour has small area, it'll be ignored
        if cv2.contourArea(c) < MinCountourArea:
            continue

        QttyOfContours = QttyOfContours + 1

        # draw an rectangle "around" the object
        (x1, y1, width, height) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), (0, 255, 0), 2)
        #
        # roi = frame[y1: y1 + height, x1: x1 + width]
        # hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
        # roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        #
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # mask = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        #
        # _, track_window = cv2.meanShift(mask, (x1, y1, width, height), term_criteria)
        # x, y, w, h = track_window
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # find object's centroid
        CoordXCentroid = (x1 + x1 + width) / 2
        CoordYCentroid = (y1 + y1 + height) / 2
        ObjectCentroid = (int(CoordXCentroid), int(CoordYCentroid))
        cv2.circle(frame, ObjectCentroid, 1, (0, 0, 255), 5)
        #
        # pts.appendleft(ObjectCentroid)

    print("Total contours found: " + str(QttyOfContours))
    cv2.putText(frame, str(QttyOfContours), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    #
    #
    # cv2.imshow("Mask", mask)
    #
    # for i in np.arange(1, len(pts)):
    #     # if either of the tracked points are None, ignore
    #     # them
    #     if pts[i - 1] is None or pts[i] is None:
    #         continue
    #
    #     # check to see if enough points have been accumulated in
    #     # the buffer
    #     if counter >= 10 and i == 1 and pts[-10] is not None:
    #         cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), 2)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(30)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()