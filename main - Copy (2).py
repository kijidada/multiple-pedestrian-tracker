import cv2
import numpy as np
import collections as col
import time;
#global variables
MaxTrackCount = 50
Buffer = 32
# MinCountourArea = 5000 #Adjust ths value according to your usage        #towncenter
# MaxCountourArea = 30000#Adjust ths value according to your usage
# Centroid_Distance = 800 #Adjust ths value according to your usage

MinCountourArea = 5000 #Adjust ths value according to your usage      #4p-c2
MaxCountourArea = 12000 #Adjust ths value according to your usage
Centroid_Distance = 150 #Adjust ths value according to your usage

# MinCountourArea = 600 #Adjust ths value according to your usage      #crosswalk
# MaxCountourArea = 10000 #Adjust ths value according to your usage
# Centroid_Distance = 1000 #Adjust ths value according to your usage
counter = 0


class ROI:
    def __init__(self, hist, roi_pos):
        self.hist = hist
        self.roi_pos = roi_pos
        self.centroid = calc_centroid(self.roi_pos)


def calc_norm_hist_frame(x, y, width, height, frame):
    roi_frame = frame[y: y + height, x: x + width]    
    hsv_roi_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv_roi_frame)
    # cv2.imshow('roi_frame', roi_frame)
    # cv2.imshow('roi_frame_h', h)
    # cv2.imshow('roi_frame_s', s)
    # cv2.imshow('roi_frame_v', v)
    # cv2.imwrite( "images/Image_"+ str(time.time())+".jpg", roi_frame )
    # cv2.imwrite( "images/h/Image_h_"+ str(time.time())+".jpg", h )
    # cv2.imwrite( "images/s/Image_s_"+ str(time.time())+".jpg", s )
    # cv2.imwrite( "images/v/Image_v_"+ str(time.time())+".jpg", v )
    roi_frame_hist = cv2.calcHist([hsv_roi_frame], [2], None, [180], [0, 180])  #cv2.calcHist([hsv_roi_frame],[0, 1], None, [180, 256], [0, 180, 0, 256]) #
    norm_roi_frame_hist = cv2.normalize(roi_frame_hist, roi_frame_hist, 0, 255, cv2.NORM_MINMAX)
    return norm_roi_frame_hist


def calc_centroid(pos):
    x, y, w, h = pos[0], pos[1], pos[2], pos[3]
    CoordXCentroid = (x + x + w) / 2
    CoordYCentroid = (y + y + h) / 2
    ObjectCentroid = (CoordXCentroid, CoordYCentroid)
    return ObjectCentroid


# video = cv2.VideoCapture("people-walking.mp4")
video = cv2.VideoCapture("4p-c2.avi")
# video = cv2.VideoCapture("crosswalk.avi")
# video = cv2.VideoCapture("TownCentreXVID.avi")

fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=80, history=50, detectShadows=0)

pts = []; #col.deque(maxlen=Buffer)
traj = []; #col.deque(maxlen=MaxTrackCount)
roi_list = []; #col.deque(maxlen=MaxTrackCount)
distance_list =[]
while True:
    _, frame = video.read()
    # frame = cv2.resize(frame.copy(), (480, 320)) 
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
    count = 0
    # check all found countours
    for c in cnts:
        count += 1 
        area = cv2.contourArea(c)
        if area < MinCountourArea:
            continue
        if area > MaxCountourArea:
            continue
        (x1, y1, width, height) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), (255, 0, 0), 2)
        # cv2.putText(frame,str(count) , (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))

        QttyOfContours = QttyOfContours + 1

        if len(roi_list) > 0:
            flag = 0
            for roi in roi_list.copy():
                d = np.linalg.norm(np.asarray(calc_centroid(roi.roi_pos)) - np.asarray( calc_centroid((x1, y1, width, height))))
                cnt_hist = calc_norm_hist_frame(x1, y1, width, height, frame)
                if d <= Centroid_Distance:
                    flag = 1
                    break
                # else:
                #     roi.hist = cnt_hist
                #     roi.roi_pos = (x1, y1, width, height)
            if flag == 0:
                new_roi = ROI(cnt_hist, (x1, y1, width, height))
                roi_list.append(new_roi)
        else:
            cnt_hist = calc_norm_hist_frame(x1, y1, width, height, frame)
            first_roi = ROI(cnt_hist, (x1, y1, width, height))
            roi_list.append(first_roi)

        # cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), (0, 0, 255), 2)
        # cv2.circle(frame, ObjectCentroid, 1, (0, 0, 255), 5)

    print("Total contours found: " + str(QttyOfContours))
    cv2.putText(frame, str(len(roi_list)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    for roi in roi_list:
        # (x1, y1, width, height) = roi.roi_pos
        # roi_hist = calc_norm_hist_frame(x1, y1, width, height, frame)
        mask = cv2.calcBackProject([hsv], [2], roi.hist, [0, 180], 1) #cv2.calcBackProject([hsv],[0,1],roi.hist,[0,180,0,256],1) # 

        # Filtering remove noise
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        # mask = cv2.filter2D(mask, -1, kernel)
        # _, mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)
        cv2.imshow("Mask", mask)
        # mask = cv2.merge((mask, mask, mask))
        # result = cv2.bitwise_and(frame, mask)
        # mask_s = cv2.resize(mask.copy(), (480, 320)) 
        # cv2.imwrite( "images/mask_s_"+ str(time.time())+str(roi_list.index(roi))+".jpg", mask_s )
        _, roi.roi_pos = cv2.meanShift(mask, roi.roi_pos, term_criteria)
        x, y, w, h = roi.roi_pos
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # find object's centroid
        roi_centroid = calc_centroid(roi.roi_pos)
        ObjectCentroid = (int(roi_centroid[0]), int(roi_centroid[1]))
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