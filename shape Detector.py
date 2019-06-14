import cv2
import numpy as np
import math
import time
import serial


Contours = {}  # define a dictionary for all contours
approx = []  # array of edges of polygon
scale = .5  # scale of the text

cap = cv2.VideoCapture(0)  # define the first cap cam
ser = serial.Serial("COM17",9600)


# fourcc = cv2.VideoWriter_fourcc(*'XVID')                                         # Define the codec and
# out = cv2.VideoWriter('output1.avi',fourcc, 20.0, (640,480))                     # create VideoWriter object

def angle(pt0, pt1, pt2):  # calculate angle
    dx1 = pt1[0][0] - pt0[0][0]
    dy1 = pt1[0][1] - pt0[0][1]
    dx2 = pt2[0][0] - pt0[0][0]
    dy2 = pt2[0][1] - pt0[0][1]
    return float((dx1 * dx2 + dy1 * dy2)) / math.sqrt(float((dx1 * dx1 + dy1 * dy1)) * (dx2 * dx2 + dy2 * dy2) + 1e-10)


time.sleep(2)
while (True):
    ret, frame = cap.read()  # Capture frame-by-frame
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                           # grayscale
        canny = cv2.Canny(frame, 80, 240, 2)                                     # Canny

        # contours
        canny2, contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(0, len(contours)):
                                                                                 # approximate the contour with accuracy
                                                                                 # proportional to the contour perimeter
            approx = cv2.approxPolyDP(contours[i], cv2.arcLength(contours[i], True) * 0.02, True)

                                                                                 # Skip small or non-convex objects
            if (abs(cv2.contourArea(contours[i])) < 100 or not (cv2.isContourConvex(approx))):
                continue
            if (len(approx) == 3):                                               # process if it triangle
                x, y, w, h = cv2.boundingRect(contours[i])
                cv2.putText(frame, 'TRI', (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 200, 255), 2, cv2.LINE_AA)
                ser.write(b'Tringle')
            elif (len(approx) >= 4 and len(approx) <= 6):
                vtc = len(approx)                                                # numb vertices of a polygonal curve
                cos = []                                                         # get cos of all corners
                for j in range(2, vtc + 1):
                    cos.append(angle(approx[j % vtc], approx[j - 2], approx[j - 1]))
                cos.sort()                                                       # sort ascending cos
                                                                                 # get lowest and highest
                mincos = cos[0]
                maxcos = cos[-1]

                x, y, w, h = cv2.boundingRect(contours[i])
                if (vtc == 4):
                    ar = w / float(h)
                    if( ar >= 0.90 and ar <= 1.1):
                        cv2.putText(frame, 'square', (x+20, y+20), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, 'RECT', (x+20, y+20), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 2, cv2.LINE_AA)
                elif (vtc == 5):
                    cv2.putText(frame, 'PENTA', (x+20, y+20), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 0), 2,
                                cv2.LINE_AA)
                elif (vtc == 6):
                    cv2.putText(frame, 'HEXA', (x+20, y+20), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
            else:
                                                                                  # detect and label circle
                area = cv2.contourArea(contours[i])
                x, y, w, h = cv2.boundingRect(contours[i])
                radius = w / 2
                if (abs(1 - (float(w) / h)) <= 2 and abs(1 - (area / (math.pi * radius * radius))) <= 0.2):
                    cv2.putText(frame, 'CIRC', (x+20, y+20), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2,cv2.LINE_AA)

#        out.write(frame)
        cv2.imshow('frame',frame)
        cv2.imshow('canny',canny)
        if cv2.waitKey(1) & 0xFF == ord('q'):                                     # it make the video window remaine
                                                                                  #  untill you press q
            break

cap.release()
cv2.destroyAllWindows()





''''        # convert image to RGB format
        q = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        canny2, contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)'''
