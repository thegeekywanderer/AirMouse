import cv2
import imutils
import numpy as np

def main():
    cap = cv2.VideoCapture(0)
    
    # ROI coordinates
    top, right, bottom, left = 30, 30, 495, 670
    
    # Initialize number of frames
    n_frame = 0
    n_image = 0

    start_recording = False

    while(True):
        (ret, frame) = cap.read()
        if(ret == True):
            # Resize and flip
            frame = imutils.resize(frame, width=700)
            frame = cv2.flip(frame, 1)

            clone = frame.copy()
            
            (height, width) = frame.shape[:2]

            roi = frame[top:bottom, right:left]
            cv2.imshow('hand nonblurred', roi)
            frame = cv2.GaussianBlur(frame, (9, 9), 0)

            cv2.imshow('Blurred', frame)
            if start_recording and n_frame % 5 == 0:
                print("writing")
                cv2.imwrite("./images/pinch_" + str(n_image) + ".png", roi)
                n_image += 1

            cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.imshow("clone", clone)
            cv2.imshow("hand", roi)
            n_frame += 1
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord('q') or n_image > 300:
            break

        if keypress == ord('s'):
            start_recording = True
        
    cap.release()
    cv2.destroyAllWindows()


main()