import cv2
import numpy as np


pt1 = (539, 215)
pt2 = (569, 270)
pt3 = (508, 270)
triangle_cnt = np.array( [pt1, pt2, pt3] )


def nothing(x):
    pass


def Trackbars():
    cv2.namedWindow('Trackbars Window')
    cv2.resizeWindow('Trackbars Window', 400,155)

    cv2.createTrackbar('Min TH','Trackbars Window',0,400,nothing)
    cv2.createTrackbar('Max TH','Trackbars Window',0,400,nothing)
    cv2.createTrackbar('S_Area','Trackbars Window',0,1000,nothing)

    cv2.setTrackbarPos('Min TH','Trackbars Window',139)
    cv2.setTrackbarPos('Max TH','Trackbars Window',240)
    cv2.setTrackbarPos('S_Area','Trackbars Window',60)


def main():
    cap = cv2.VideoCapture(0)

    while True:
        counter = {"TRI":0, "Circ":0, "Sqr":0, "Rec":0}
        ret, frame = cap.read()
        Min = cv2.getTrackbarPos('Min TH','Trackbars Window')
        Max = cv2.getTrackbarPos('Max TH','Trackbars Window')
        SA = cv2.getTrackbarPos('S_Area','Trackbars Window')

        canny = cv2.Canny(frame,Min,Max)
        contours, hierarchy = cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)
        for i in range (len(contours)):
            cnt = contours[i]
            primeter = cv2.arcLength(cnt,True)
            epsilon = 0.03*primeter
            approx = cv2.approxPolyDP(cnt,epsilon,True)

            if (abs(cv2.contourArea(contours[i])) < SA ):
                continue

            if len(approx)==2:
                counter["Rec"] += 1

            if len(approx)== 3:
                counter["TRI"] += 1

            if len(approx)==4:
                x,y,w,h=cv2.boundingRect(cnt)
                aspect_ratio = float(w)/h
                if aspect_ratio>0.9 and aspect_ratio<1.1:
                    counter["Sqr"] += 1

                else:
                    counter["Rec"] += 1

            if len(approx)>=7 :
                counter["Circ"] += 1


        cir = int(counter["Circ"] * 0.5)
        sqr = int(counter["Sqr"] * 0.5)
        rec = int(counter["Rec"] * 0.5)
        tri = int(counter["TRI"] * 0.5)


        cv2.rectangle(frame,(509,298),(567,291),(0,0,255),-1)
        cv2.rectangle(frame,(564,451),(510,398),(0,0,255),-1)
        cv2.circle(frame,(537,348),32,(0,0,255),-1)
        cv2.drawContours(frame, [triangle_cnt], 0, (0,0,255), -1)

        cv2.putText(frame,'{}'.format(cir),(587,368), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame,'{}'.format(sqr),(587,443), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame,'{}'.format(rec),(587,303), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame,'{}'.format(tri),(587,253), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('canny',canny)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

Trackbars()
main()