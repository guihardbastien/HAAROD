import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

test=0

while 1:
        
    ret, img = cap.read()
    img2=cv2.flip(img,1)
    
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img2,str(test),(10,250), font, 0.5, (11,255,255), 1, cv2.LINE_AA)
    cv2.rectangle(img2,(600,150),(500,350),(255,255,255),2)

    test = 'no faces detected'

    _, frame = cap.read()
    hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([45,100,100])
    upper_red = np.array([78,255,255])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(img2,img2, mask= mask)

    mask2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(img2, contours, -1, (0,0,255), 3)
    
    for contour in contours:
    
        [x,y,w,h] = cv2.boundingRect(contour)
        
        cv2.putText(img2,'+',(x,y-5), font, 0.8, (11,255,255), 1, cv2.LINE_AA)

        if x>500 and x<600 and y>150 and y<350:
            cv2.putText(img2,'TEST',(20,130), font, 4, (11,255,255), 1, cv2.LINE_AA)
       
    for (x,y,w,h) in faces:
        
        cv2.putText(img2,'VISAGE',(x,y-5), font, 0.8, (11,255,255), 1, cv2.LINE_AA)
       
        cv2.rectangle(img2,(x,y),(x+w,y+h),(255,0,0),1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        test = (x,y)

    cv2.imshow('img',img2)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
 
cap.release()
cv2.destroyAllWindows()
