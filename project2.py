'''
=================================
|  Project multimediaverwerking |
=================================
Omschrijving
----------
TODO


Gebruik
-----
project2.py


Keys
----
Q - exit
'''
# Python 2/3 compatibility
from __future__ import print_function

import cv2
import numpy as np
#from common import anorm2, draw_str
from time import clock



lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )


class App:
    def __init__(self, video_src):
        self.cap = cv2.VideoCapture('frere.mp4')
        #print self.cap.get(cv2.CAP_PROP_FPS)
        # Mixture of gradients
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        
    def run(self):
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            resize_factor = 0.45
            frame = cv2.resize(frame, (0,0), fx=resize_factor, fy=resize_factor)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frame = self.fgbg.apply(frame_gray, learningRate = 0.001)
            kernel = np.ones((5,5),np.uint8)
            frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
            frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
            #frame = cv2.medianBlur(frame,3)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    

def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    print(__doc__)
    App(video_src).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
