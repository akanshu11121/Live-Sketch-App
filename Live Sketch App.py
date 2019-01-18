
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import time


# In[5]:


cam = cv2.VideoCapture(0)

while True:
    ret,frame = cam.read()
    frame = cv2.flip(frame,1)
    
    canvas = np.zeros((frame.shape[0],frame.shape[1]))
    page = np.ones((frame.shape[0],frame.shape[1]))
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    canny = cv2.Canny(gray,70,130)
    
    image,contour,heirarichy = cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
    brush = sorted(contour,key=cv2.contourArea,reverse=True)
        
    for c in contour:
        approx = 0.005 * cv2.arcLength(c,True)
        accuracy = cv2.approxPolyDP(c,approx,True)
        sketch = cv2.drawContours(canvas,[accuracy],-1,(230,230,230),1)
        draw = cv2.drawContours(page,[accuracy],-1,(0,0,0),1)
        if cv2.waitKey(1) & 0xFF == ord('k'):
            break
        cv2.imshow('sketch',sketch)
        cv2.imshow('draw',draw)
        
    time.sleep(3)    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

