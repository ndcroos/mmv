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
#from __future__ import print_function

import cv2
import numpy as np
from common import anorm2, draw_str
from time import clock



lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Create some random colors
color = np.random.randint(0,255,(100,3))
class Entity(object):
    staticID = 0
    def __init__(self, pos):
        self.pos = [list(pos)]
        self.lastSeen = 0
        self.ID = Entity.staticID
        Entity.staticID += 1
        self.age = 0
        
    def move(self, pos):
        self.age += 1
        self.pos.append(list(pos))
    
    def pop(self):
        self.age -= 1
        self.pos.pop()
                       
                       
class App:
    def __init__(self, video_src):
        self.cap = cv2.VideoCapture(video_src)
        #print self.cap.get(cv2.CAP_PROP_FPS)
        # Mixture of gradients
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.median = None
        self.clahe_median = None
        self.people_IN = 0
        self.people_OUT = 0
        self._IN = 0
        self.people_OUT = 0
        # LK
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.frame_idx = 0
        
    def preprocess(self):
        frames = []
        # _: om aan te tonen dat index hier niet gebruikt wordt
        #preprocess x video framewes
        for _ in range(400):
            ret, frame = self.cap.read();
            resize_factor = 0.45
            frame = cv2.resize(frame, (0,0), fx=resize_factor, fy=resize_factor)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        self.median = np.median(frames, axis=0).astype(dtype=np.uint8)
        cv2.imshow('preprocess', self.median)
        # self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, framePOSITIE)
        self.clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(8,8))
        self.clahe_median = self.clahe.apply(self.median)
        cv2.imshow('clahe_median', self.clahe_median)
        
    def run(self):
        
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            resize_factor = 0.45
            frame = cv2.resize(frame, (0,0), fx=resize_factor, fy=resize_factor)
            original = frame.copy()
            cv2.imshow('original', original)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            
            gray = self.fgbg.apply(gray, learningRate = 0.001)
            kernel = np.ones((5,5),np.uint8)
            gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            vis = gray.copy()
            #frame = cv2.medianBlur(frame,3)
            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])
                        
            self.frame_idx += 1
            self.prev_gray = gray
            cv2.imshow('lk_track', vis)
            
            cv2.imshow('frame', gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()
    

def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    print(__doc__)
    a = App('Nederkouter.mp4')
    a.preprocess()
    a.run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
