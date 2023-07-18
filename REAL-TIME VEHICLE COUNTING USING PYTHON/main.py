import cv2
import numpy as np
from time import sleep

width_min=80 #Minimum width of the rectangle
height_min=80 #Minimum height of the rectangl

offset=6 #Allowed error between pixels

pos_line=550 #Position( Y cordinates) of the counting line

fps= 60 #Video frames per second

detect = []
counts = 0

#This function takes the x, y, width, and height of a rectangle as
# input and returns the coordinates of the center of the rectangle
def center_rec(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

cap = cv2.VideoCapture('video.mp4') #the VideoCapture object from OpenCV.
subtracao = cv2.bgsegm.createBackgroundSubtractorMOG() #The subtracao variable is initialized with a background subtractor created using the Mixture
# of Gaussians `(MOG) algorithm`.



while True:
    ret , frame = cap.read()
    tempo = float(1/fps)
    sleep(tempo)
    #This initiates an infinite loop to process each frame of the video. cap.read() reads the next frame from the video capture,
    # and the sleep function introduces a delay between frames to achieve the desired frame rate.
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = subtracao.apply(blur)       #These lines perform preprocessing on the frame to enhance object detection.
    # The frame is converted to grayscale (grey), blurred using a Gaussian filter (blur),
    # and then passed through the background subtractor (subtracao) to extract moving objects (img_sub).
    # The resulting image is dilated and morphologically closed to enhance object shapes and remove noise.
    dilated = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx (dilated, cv2. MORPH_CLOSE , kernel)
    dilatada = cv2.morphologyEx (dilatada, cv2. MORPH_CLOSE , kernel)
    contorno,h=cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame, (25, pos_line), (1200, pos_line), (255,127,0), 3)
    for(i,c) in enumerate(contorno):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_contorno = (w >= width_min) and (h >= height_min)
        if not validate_contorno:
            continue

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        center = center_rec(x, y, w, h)
        detect.append(center)
        cv2.circle(frame, center, 4, (0, 0,255), -1)

        for (x,y) in detect:
            if y<(pos_line+offset) and y>(pos_line-offset):
                counts+=1
                cv2.line(frame, (25, pos_line), (1200, pos_line), (0,127,255), 3)
                detect.remove((x,y))
                print("car is detected : "+str(counts))
       
    cv2.putText(frame, "VEHICLE COUNT : "+str(counts), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    cv2.imshow("Video Original" , frame)
    cv2.imshow("Detectar",dilatada)

    if cv2.waitKey(1) == 27:
        break
    
cv2.destroyAllWindows()
cap.release()
