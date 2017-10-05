import time

import cv2

from pedestrian_lib_r import *

from picamera.array import PiRGBArray
from picamera import PiCamera

camera = PiCamera()
camera.resolution = (640,480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640,480))

time.sleep(0.1)

def process(cv_image, cascade):
##        cv_image = cv2.imread(frame)

        haar_objects = detect_objects(cv_image, cascade)
        color_blobs = detect_blobs(cv_image)
        final_blobs = []

        for haar_object in haar_objects:
            found = False
            for color_blob in color_blobs:
                if is_overlapping(haar_object, color_blob):
                    found = True
                    break
            if found:
                final_blobs.append(haar_object)

        print (final_blobs)
        if (len(final_blobs)!=0):
                print ("Pedestrian Crossing")


        return show_blobs(cv_image, final_blobs)

if __name__ == '__main__':

    cap = cv2.VideoCapture(0)

    while True:
        for imageFrame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            frame = imageFrame.array

##            ret, frame = cap.read()
        
            classifier_filename = "haar_classifier.xml"

            cascade = cv2.CascadeClassifier(classifier_filename)
    ##        print frame

            cv2.imshow('frame',process(frame, cascade))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


