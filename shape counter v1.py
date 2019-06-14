import cv2
import numpy as np
import math
import time
import serial

Contours = {} 
approx = [] 
scale = .5  
ser = serial.Serial("COM16",9600)

cap = cv2.VideoCapture(0)

counter ={}

def angle(pt0, pt1, pt2): 
    dx1 = pt1[0][0] - pt0[0][0]
    dy1 = pt1[0][1] - pt0[0][1]
    dx2 = pt2[0][0] - pt0[0][0]
    dy2 = pt2[0][1] - pt0[0][1]
    return float((dx1 * dx2 + dy1 * dy2)) / math.sqrt(float((dx1 * dx1 + dy1 * dy1)) * (dx2 * dx2 + dy2 * dy2) + 1e-10)


time.sleep(2)
shape=["TRI","C","Sqr","Rec"]
while (True):
    ret, frame = cap.read() 
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                          
        canny = cv2.Canny(frame, 80, 240, 2)                                     

        # contours
    for shap in shape:
        counter[shap]=0
                              
                                            
        cv2.imshow('frame',frame)
        cv2.imshow('canny',canny)

        if cv2.waitKey(1) & 0xFF == ord('o'):

            cv2.imwrite("frame.jpg", frame)         
            image = cv2.imread("frame.jpg")
            gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                   
            canny1 = cv2.Canny(image, 80, 240, 2)
            circles = cv2.HoughCircles(gray1, cv2.HOUGH_GRADIENT, 1.2, 750)
            no_of_circles = 0
            if circles is not None:
                        circles = np.round(circles[0, :]).astype("int")
                        no_of_circles = len(circles)

            contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for i in range(0, len(contours)):
                        approx = cv2.approxPolyDP(contours[i], cv2.arcLength(contours[i], True) * 0.02, True)
                        if (abs(cv2.contourArea(contours[i])) < 100 or not (cv2.isContourConvex(approx))):
                            continue

                        if (len(approx) == 3):                                             
                            x, y, w, h = cv2.boundingRect(contours[i])
                            cv2.putText(image, 'TRI', (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 200, 255), 2, cv2.LINE_AA)
                            ser.write(b'Tringle ')

                            shap="TRI"
                            counter[shap]+=1
                        elif (len(approx) >= 4):
                            vtc = len(approx)                                               
                            cos = []                                                        
                            for j in range(2, vtc + 1):
                                cos.append(angle(approx[j % vtc], approx[j - 2], approx[j - 1]))
                            cos.sort()                                                      
                                                                                            
                            mincos = cos[0]
                            maxcos = cos[-1]

                            x, y, w, h = cv2.boundingRect(contours[i])
                            if (vtc == 4):
                                ar = w / float(h)
                                if( ar >= 0.85 and ar <= 1.2):
                                    cv2.putText(image, 'square', (x+20, y+20), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 2, cv2.LINE_AA)
                                    ser.write(b'Square ')

                                    shap="Sqr"
                                    counter[shap]+=1


                                else:
                                    cv2.putText(image, 'RECT', (x+20, y+20), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 2, cv2.LINE_AA)
                                    shap="Rec"
                                    ser.write(b'Rectangular ')

                                    counter[shap]+=1

                        elif (len(approx) <= 5):

                            shap="C"
                            counter[shap]+=1

                   

                            
            print("tri",counter["TRI"])
            print("sqr",counter["Sqr"])
            print("rec",counter["Rec"])
            print("cir",counter["C"])
            print( 'no of circles',no_of_circles)

        if cv2.waitKey(1) & 0xFF == ord('q'):                                   
                                                                                 
            break

cap.release()
cv2.destroyAllWindows()

