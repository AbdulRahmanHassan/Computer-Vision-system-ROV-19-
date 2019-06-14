import cv2                          #call opencv library for dealing with camera
import numpy as np                  #call numpy library
import math                         #call math library for angle calculation
import serial                       #call serial library to send result to arduino


ser = serial.Serial("COM16",9600)   #define port (com) 16 as serial port with 9600 buadrate
Contours = {}                       #define tuple to locate the countour
approx = []                         
cap = cv2.VideoCapture(0)           #define camera number 0 as a main camer


def angle(pt0, pt1, pt2): 
    dx1 = pt1[0][0] - pt0[0][0]
    dy1 = pt1[0][1] - pt0[0][1]
    dx2 = pt2[0][0] - pt0[0][0]
    dy2 = pt2[0][1] - pt0[0][1]
    return float((dx1 * dx2 + dy1 * dy2)) / math.sqrt(float((dx1 * dx1 + dy1 * dy1)) * (dx2 * dx2 + dy2 * dy2) + 1e-10)


while True:
    ret, frame = cap.read()         #Capture frame-by-frame form the main camera
    if ret == True:
        canny = cv2.Canny(frame, 80, 240, 2)                                             
                                            
    cv2.imshow('frame',frame)       #Show viedo captured

    if cv2.waitKey(1) & 0xFF == ord('o'):   #wait key 'o' to perform the program

            cv2.imwrite("frame.jpg", frame) #save the current image from the video to deal with it   
            image = cv2.imread("frame.jpg") #read that image

            #apply canny filter to the image                   
            canny1 = cv2.Canny(image, 80, 240, 2)

            contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)           #find the contours
            for i in range(0, len(contours)):
                        approx = cv2.approxPolyDP(contours[i], cv2.arcLength(contours[i], True) * 0.02, True)
                        if (abs(cv2.contourArea(contours[i])) < 100 or not (cv2.isContourConvex(approx))):   
                            continue                 #ignore small areas to not count in the program 

                        if (len(approx) == 3):       #if number of sides equal 3 then it will be trignle                                     
                            ser.write(b'Tringle ')   #send string tringle as bits

                        elif (len(approx) >= 4):
                            vtc = len(approx)        #number vertices of a polygonal curve
                            cos = []                 #get cos of all corners                                       
                            for j in range(2, vtc + 1):
                                cos.append(angle(approx[j % vtc], approx[j - 2], approx[j - 1]))
                            cos.sort()               #sort ascending cos  
                                                                                            
                            mincos = cos[0]
                            maxcos = cos[-1]

                            x, y, w, h = cv2.boundingRect(contours[i])
                            if (vtc == 4):
                                ar = w / float(h)    #define ratio of width to heigh 
                                if( ar >= 0.85 and ar <= 1.2):        #if ratio in range of 0.85 to 1.2 
                                    ser.write(b'Square ')             # the shape while be square


                                else:                                 #else the shape is rectangular
                                    ser.write(b'Rectangular ')

                        if (len(approx) == 5):       #if number of sides equal 5 then it will be trignle                                     
                            ser.write(b'Pentagon ')   #send string Pentagon as bits

                        if (len(approx) == 6):       #if number of sides equal 6 then it will be trignle                                     
                            ser.write(b'Hexagonal ')   #send string Hexagonal as bits

                        if (len(approx)>6):
                            ser.write(b'Circule ')


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()

