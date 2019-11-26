import cv2
import numpy as np
import math
from numpy import array
import collections as col
import time;

print(cv2.__version__)
#global variables
MaxTrackCount = 50
Buffer = 20
dT = 50
frameAdvance =150
counter = 0
(dX, dY) = (0, 0)

lastCentroidList = []
lastPositionList = []
lastdirectionList = []
IDList = []
trackedROI = []
# MinCountourArea = 800 #Adjust ths value according to your usage        #towncenter
# MaxCountourArea = 3000#Adjust ths value according to your usage
# Centroid_Distance = 800 #Adjust ths value according to your usage

MinCountourArea = 3500 #Adjust ths value according to your usage      #4p-c2
MaxCountourArea = 14000 #Adjust ths value according to your usage
Centroid_Distance = 150 #Adjust ths value according to your usage
collision_distance = 100 #Adjust ths value according to your usage

# MinCountourArea = 200 #Adjust ths value according to your usage      #crosswalk
# MaxCountourArea = 10000 #Adjust ths value according to your usage
# Centroid_Distance = 1000 #Adjust ths value according to your usage


class ROI:
    def __init__(self, hist, roi_pos, ID):
        self.ID = ID
        self.hist = hist
        self.roi_pos = roi_pos
        self.centroid = []
        self.velocity = []
        self.acceleration = []
        self.direction = -1
        self.proj_pos = (0,0)
        self.motionTrackOn = 0
        self.neighbourhood = []
        self.frmCnt = 0


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
    # mask_frame = cv2.inRange(hsv_roi_frame, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    # cv2.imwrite( "images/Image_mask_frame_"+ str(time.time())+".jpg", mask_frame )
    roi_frame_hist = cv2.calcHist([hsv_roi_frame], [2], None, [180], [0, 180])  # 4p-c2
    # roi_frame_hist = cv2.calcHist([hsv_roi_frame],[0, 1], None, [180, 256], [0, 180, 0, 256]) #towncenter
    norm_roi_frame_hist = cv2.normalize(roi_frame_hist, roi_frame_hist, 0, 255, cv2.NORM_MINMAX)
    return norm_roi_frame_hist

def calc_centroid(pos):
    x, y, w, h = pos[0], pos[1], pos[2], pos[3]
    CoordXCentroid = (x + x + w) / 2
    CoordYCentroid = (y + y + h) / 2
    ObjectCentroid = (CoordXCentroid, CoordYCentroid)
    return ObjectCentroid

def calc_roi_projection(roi, t=frameAdvance):
    velocity = []
    displacement = []
    centroids = roi.centroid
    f = []
    for index,item in enumerate(centroids): 
        if index == len(centroids) -10:
            break
        displacement.append((centroids[-1-index][0] - centroids[-10-index][0],centroids[-1-index][1] - centroids[-10-index][1]))
        velocity.append(((centroids[-1-index][0] - centroids[-10-index][0])/dT, (centroids[-1-index][1] -centroids[-10-index][1])/dT))
    f.append((velocity[-1][0] - velocity[-2][0])/dT)
    f.append((velocity[-1][0] - velocity[-2][0])/dT)
    roi.velocity = velocity[-1]
    roi.acceleration = f[-1]
    # cv2.putText(frame, "dx: {}, dy: {}".format(displacement[-1][0], displacement[-1][1]),
	# 	(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
	# 	0.35, (0, 0, 255), 1)
    roi.direction = get_roi_direction_from_displacement(displacement[-1])
    proj_point = project_point(velocity[-1], f, t, centroids[-1])
    roi.proj_pos = proj_point
    return (int(proj_point[0]),int(proj_point[1]))

def get_roi_direction_from_displacement(displacement):
    direction = -1
    dirX = -1
    dirY = -1
    dX = displacement[0]
    dY = displacement[1]
    if np.abs(dX) > 5:
        dirX = 0 if np.sign(dX) == 1 else 4
			# ensure there is significant movement in the
			# y-direction
    if np.abs(dY) > 5:
        dirY = 6 if np.sign(dY) == 1 else 2
			# handle when both directions are non-empty
    if dirX != -1 and dirY != -1:
        if dirX == 0 and dirY == 2:
            direction = 1
        elif dirX == 0 and dirY == 6:
            direction = 7
        elif dirX == 4 and dirY == 6:
            direction = 5
        elif dirX == 4 and dirY == 2:
            direction = 3
    else:
        direction = dirX if dirX != -1 else dirY
    return direction

def project_point(v, f, t, point):
    projected_point = point
    # projected_point = (point[0] + v[0]*t + 0.5*f[0]*t**2, point[1] + v[1]*t + 0.5*f[1]*t**2)
    projected_point = (point[0] + v[0]*t, point[1] + v[1]*t )
    return projected_point

def generate_bbox_from_centroid(centroid, w,h):
    return (centroid[0],centroid[1],centroid[0]+w,centroid[1]+h)

def detect_collision(roi):
    collision_bbox_id = []
    for index, ctrd in enumerate(lastCentroidList):
        dist = np.linalg.norm(np.asarray(calc_centroid(roi.roi_pos)) - np.asarray(lastCentroidList[index]))         
        if roi.motionTrackOn == 1:
            if dist > collision_distance and roi.neighbourhood[-1] == IDList[index]:
                roi.motionTrackOn = 0
                continue
        if IDList[index] != roi.ID and roi.motionTrackOn == 0:
            if dist <= collision_distance :
                if lastdirectionList[index] == (roi.direction + 4) % 8:
                    collision_bbox_id.append((IDList[index], dist))
                    roi.frmCnt = int((dist/math.sqrt(roi.velocity[0]**2 + roi.velocity[1]**2))/dT)
                    roi.motionTrackOn = 1
                    roi.neighbourhood.append(IDList[index])
                    break
    return collision_bbox_id

def predict_bbox(roi, dist=0):
    (x,y,w,h) = roi.roi_pos
    if roi.frmCnt > 0 :
        (x,y,w,h) = generate_bbox_from_centroid(calc_roi_projection(roi,dT),roi.roi_pos[2],roi.roi_pos[3])
        roi.frmCnt = roi.frmCnt - 1
    return (x,y,w,h)




# video = cv2.VideoCapture("people-walking.mp4")
video = cv2.VideoCapture("4p-c2.avi")
# video = cv2.VideoCapture("crosswalk.avi")
# video = cv2.VideoCapture("TownCentreXVID.avi")
# video = cv2.VideoCapture("campus7-c2.avi")

# fgbg = cv2.createBackgroundSubtractorKNN( history=50, detectShadows=0)
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=80, history=50, detectShadows=1)

pts = []; #col.deque(maxlen=Buffer)
traj = []; #col.deque(maxlen=MaxTrackCount)
roi_list = []; #col.deque(maxlen=MaxTrackCount)
distance_list =[]
while True:
    # lastCentroidList = []
    # lastPositionList = []
    # lastdirectionList = []
    # IDList = []
    _, frame = video.read()
    # frame = cv2.resize(frame.copy(), (480, 320)) 
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    fgmask = fgbg.apply(blur)
    cv2.imshow('fgmask', fgmask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    dilation = cv2.dilate(fgmask, kernel, iterations=3)
    # opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    # ckernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # eroded = cv2.erode(dilation, ckernel, iterations=5)
    cv2.imshow('dilation', dilation)
    _, cnts, _ = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    QttyOfContours = 0

    term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    count = 0
    # check all found countours
    for index, c in enumerate(cnts):
        count += 1 
        area = cv2.contourArea(c)
        
        if area < MinCountourArea:
            continue
        if area > MaxCountourArea:
            continue

        (x1, y1, width, height) = cv2.boundingRect(c)
        ctrd = calc_centroid((x1, y1, width, height))
        # cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), (255, 0, 0), 2)
        # cv2.putText(frame,str(area)+"__"+str(count) , (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
        QttyOfContours = QttyOfContours + 1

        if len(roi_list) > 0:
            flag = 0
            for roi in roi_list.copy():
                d = np.linalg.norm(np.asarray(calc_centroid(roi.roi_pos)) - np.asarray(ctrd))
                cnt_hist = calc_norm_hist_frame(x1, y1, width, height, frame)
                if d <= Centroid_Distance:
                    flag = 1
                    break
                # else:
                    # roi_frame = frame[roi.roi_pos[1]: roi.roi_pos[1] + roi.roi_pos[3], roi.roi_pos[0]: roi.roi_pos[0] + roi.roi_pos[2]]    
                    # hsv_roi_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
                    # h,s,v = cv2.split(hsv_roi_frame)
                    # cv2.imwrite( "images/Image_"+ str(time.time())+".jpg", roi_frame )
                    # cv2.imwrite( "images/h/Image_h_"+ str(time.time())+".jpg", h )
                    # cv2.imwrite( "images/s/Image_s_"+ str(time.time())+".jpg", s )
                    # cv2.imwrite( "images/v/Image_v_"+ str(time.time())+".jpg", v )                  
                    # roi.hist = calc_norm_hist_frame(roi.roi_pos[0],roi.roi_pos[1],roi.roi_pos[2],roi.roi_pos[3], frame)
                    # roi.roi_pos = (x1, y1, width, height)
            if flag == 0:
                new_roi = ROI(cnt_hist, (x1, y1, width, height),index)
                roi_list.append(new_roi)
        else:
            cnt_hist = calc_norm_hist_frame(x1, y1, width, height, frame)
            first_roi = ROI(cnt_hist, (x1, y1, width, height),index)
            roi_list.append(first_roi)

    print("Total contours found: " + str(QttyOfContours))
    cv2.putText(frame, str(len(roi_list)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    # cv2.putText(frame, str(cv2.CAP_PROP_FPS), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    for roi in roi_list:
        col_roi = detect_collision(roi)
        if(roi.motionTrackOn == 0):
            mask = cv2.calcBackProject([hsv], [2], roi.hist, [0, 180], 1) # 4p-c2
            # mask = cv2.calcBackProject([hsv],[0,1],roi.hist,[0,180,0,256],1) #towncenter
            _, roi.roi_pos = cv2.meanShift(mask, roi.roi_pos, term_criteria)
            x, y, w, h = roi.roi_pos
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            x, y, w, h = predict_bbox(roi)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
        # find object's centroid
        roi_centroid = calc_centroid(roi.roi_pos)
        ObjectCentroid = (int(roi_centroid[0]), int(roi_centroid[1]))
        if len(roi.centroid) == Buffer:
            del roi.centroid[0]
        roi.centroid.append(ObjectCentroid)
        cv2.imshow("Mask: "+str(roi.ID), mask)
        if len(roi.centroid) >= Buffer:
            proj_pt = calc_roi_projection(roi)
            cv2.putText(frame,str(roi.direction) , (x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
            cv2.circle(frame, proj_pt, 1, (0, 0, 255), 2)
        
        for index, item in enumerate(roi.centroid): 
            if index == len(roi.centroid) -1:
                break
            cv2.line(frame, roi.centroid[index], roi.centroid[index + 1], [255, 0, 255], 2) 
        cv2.putText(frame,str(roi.ID) , (x+w,y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
        # img = np.zeros(frame.shape, np.uint8)
        # a = np.asarray(roi.centroid)
        # cv2.drawContours(img, [a], 0, (100,100,255), 2)
        if roi.ID not in IDList:
            IDList.append(roi.ID)
            lastCentroidList.append(roi.centroid[-1])
            lastPositionList.append(roi.roi_pos)
            lastdirectionList.append(roi.direction)
        else:
            lastCentroidList[IDList.index(roi.ID)] = roi.centroid[-1]
            lastPositionList[IDList.index(roi.ID)]=roi.roi_pos
            lastdirectionList[IDList.index(roi.ID)] = roi.direction
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(dT)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()