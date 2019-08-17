import cv2
import  numpy as np
import time

def nothing(x):
    pass

cv2.namedWindow('Trackbars Window')
cv2.resizeWindow('Trackbars Window', 400, 155)
cv2.createTrackbar('S_Area', 'Trackbars Window', 0, 1000, nothing)
cv2.setTrackbarPos('S_Area', 'Trackbars Window', 30)

cap = cv2.VideoCapture(0)
lline = None
while True:
    ret, frame = cap.read()
    SA = cv2.getTrackbarPos('S_Area', 'Trackbars Window')

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([110, 100, 100])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    blur0 = cv2.GaussianBlur(res,(5,5),0)
    blur = cv2.GaussianBlur(blur0,(5,5),0)

    canny = cv2.Canny(blur,140,240)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)

    for i in range (len(contours)):
        cnt = contours[i]
        if (abs(cv2.contourArea(contours[i])) < SA):
            continue
        else:
            x,y,w,h = cv2.boundingRect(cnt)
            px_cm = (float(h)/1)
            line = (float(w)/px_cm)
            lline = '{0:.2f}'.format(line)

    cv2.putText(frame, '{} cm'.format(lline), (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
