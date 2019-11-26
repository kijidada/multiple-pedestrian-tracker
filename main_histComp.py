import cv2
import numpy as np
import collections as col
#global variables
MaxTrackCount = 50
Buffer = 32
MinCountourArea = 6000  #Adjust ths value according to your usage
MaxCountourArea = 9000 #Adjust ths value according to your usage
SimilarityMeasure = 0.4 #Adjust ths value according to your usage
counter = 0


class ROI:
    def __init__(self, hist, roi_pos, centroid):
        self.hist = hist
        self.roi_pos = roi_pos
        self.centroid = centroid

def point_inside_polygon(x_p, y_p, poly):

    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for j in range(n+1):
        p2x, p2y = poly[j % n]
        if y_p > min(p1y, p2y):
            if y_p <= max(p1y, p2y):
                if x_p <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y_p-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x_p <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def calc_norm_hist_frame(x, y, width, height, frame):
    roi_frame = frame[y: y + height, x: x + width]
    cv2.imshow('roi_frame', roi_frame)
    hsv_roi_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    roi_frame_hist = cv2.calcHist([hsv_roi_frame], [0], None, [180], [0, 180])
    norm_roi_frame_hist = cv2.normalize(roi_frame_hist, roi_frame_hist, 0, 255, cv2.NORM_MINMAX)
    return norm_roi_frame_hist


# video = cv2.VideoCapture("people-walking.mp4")
# video = cv2.VideoCapture("4p-c2.avi")
# video = cv2.VideoCapture("crosswalk.avi")
video = cv2.VideoCapture("TownCentreXVID.avi")

fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=80, history=50, detectShadows=0)

pts = []; #col.deque(maxlen=Buffer)
traj = []; #col.deque(maxlen=MaxTrackCount)
roi_list = []; #col.deque(maxlen=MaxTrackCount)
while True:
    _, frame = video.read()
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    fgmask = fgbg.apply(blur)
    cv2.imshow('fgmask', fgmask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    dilation = cv2.dilate(fgmask, kernel, iterations=4)
    # opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    # ckernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # eroded = cv2.erode(dilation, ckernel, iterations=5)
    cv2.imshow('dilation', dilation)
    _, cnts, _ = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    QttyOfContours = 0

    term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    # check all found countours
    for c in cnts:
        # if a contour has small area, it'll be ignored
        area = cv2.contourArea(c)
        if area < MinCountourArea:
            continue
        if area > MaxCountourArea:
            continue

        (x1, y1, width, height) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), (255, 0, 0), 2)

        if len(roi_list) > 0:
            for roi in roi_list:
                cnt_hist = calc_norm_hist_frame(x1, y1, width, height, frame)
                d = cv2.compareHist(roi.hist, cnt_hist, cv2.HISTCMP_CORREL)
                if d > SimilarityMeasure:
                    flag = 1
                    break
                else:
                    new_roi = ROI(cnt_hist, (x1, y1, width, height))
                    roi_list.append(new_roi)
                    QttyOfContours = QttyOfContours + 1
        else:
            cnt_hist = calc_norm_hist_frame(x1, y1, width, height, frame)
            first_roi = ROI(cnt_hist, (x1, y1, width, height))
            roi_list.append(first_roi)
            QttyOfContours = QttyOfContours + 1

        # cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), (0, 0, 255), 2)
        # cv2.circle(frame, ObjectCentroid, 1, (0, 0, 255), 5)

    print("Total contours found: " + str(QttyOfContours))
    cv2.putText(frame, str(QttyOfContours), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    for roi in roi_list:
        # (x1, y1, width, height) = roi.roi_pos
        # roi_hist = calc_norm_hist_frame(x1, y1, width, height, frame)
        mask = cv2.calcBackProject([hsv], [0], roi.hist, [0, 180], 1)
        cv2.imshow("Mask", mask)
        _, roi.roi_pos = cv2.meanShift(mask, roi.roi_pos, term_criteria)
        x, y, w, h = roi.roi_pos
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # find object's centroid
        CoordXCentroid = (x + x + w) / 2
        CoordYCentroid = (y + y + h) / 2
        ObjectCentroid = (int(CoordXCentroid), int(CoordYCentroid))
        cv2.circle(frame, ObjectCentroid, 1, (0, 0, 255), 5)

    #
    #
    # cv2.imshow("Mask", mask)

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