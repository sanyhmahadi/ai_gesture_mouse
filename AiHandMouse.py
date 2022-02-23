import cv2
import HandTrackingModule as Htm
import time
import autopy
import math
import mediapipe as mp
import numpy as np


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, modelComplexity=1, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils  # it gives small dots on hands total 20 landmark points
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        # Send rgb image to hands
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)  # process the frame
        #     print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    # Draw dots and connect them
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):

        xList = []
        yList = []
        bbox = []
        self.lmlist = []
        # check wether any landmark was detected
        if self.results.multi_hand_landmarks:
            # Which hand are we talking about
            myHand = self.results.multi_hand_landmarks[handNo]
            # Get id number and landmark information
            for id, lm in enumerate(myHand.landmark):
                # id will give id of landmark in exact index number
                # height width and channel
                h, w, c = img.shape
                # find the position
                cx, cy = int(lm.x * w), int(lm.y * h)  # center
                # print(id,cx,cy)
                xList.append(cx)
                yList.append(cy)
                self.lmlist.append([id, cx, cy])
                # Draw circle for 0th landmark
                if draw:
                    cv2.circle(img, (cx, cy), 15, (128, 0, 128), 1)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (255, 0, 128), 2)

        return self.lmlist, bbox

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmlist[self.tipIds[0]][1] > self.lmlist[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):

            if self.lmlist[self.tipIds[id]][2] < self.lmlist[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # totalFingers = fingers.count(1)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmlist[p1][1:]
        x2, y2 = self.lmlist[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    # Frame rates
    pTime = 0
    cTime = 0
    # cap = cv2.VideoCapture(0)
    # detector = handDetector()
    # while True:
    #     success,img = cap.read()
    #     img = detector.findHands(img)
    #     lmlist = detector.findPosition(img)
    #     if len(lmlist) != 0:
    #         #print(lmlist[4])
    #
    #         cTime = time.time()
    #         fps = 1/(cTime-pTime)
    #         pTime = cTime
    #
    #
    #     image = cv2.flip(img, flipCode=1)
    #     cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    #     cv2.imshow("Video", image)
    #     if cv2.waitKey(1) == ord('q'):
    #         break
    #
    # cap.release()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

wCam, hCam = 640, 480
frameR = 100  # frame reduction
smootherThing = 7

pTime = 0
plockX, plockY = 0, 0
clockX, clockY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
ptime = 0
detector = handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
# print(wScr, hScr)

while True:
    # find the hand landmark
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist, bbox = detector.findPosition(img)
    # get the tip of index and middle fingers
    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]
        # print(x1, y1, x2, y2)

        # check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (128, 255, 255), 3)
        # Only index finger : Moving Finger
        if fingers[1] == 1 and fingers[2] == 0:
            # convert coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            # smoothing values
            clockX = plockX + (x3 - plockX) / smootherThing
            clockY = plockY + (y3 - plockY) / smootherThing
            # Move Mouse
            autopy.mouse.move(wScr - clockX, clockY)
            cv2.circle(img, (x1, y1), 15, (190, 180, 169), cv2.FILLED)
            plockX, plockY = clockX, clockY
        # Clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            # checking the distance
            length, img, lineinfo = detector.findDistance(8, 12, img)
            print(length)
            # click mouse if distance short
            if length < 30:
                cv2.circle(img, (lineinfo[4], lineinfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()
    # frame rate
    cTime = time.time()
    fps = 1 / (cTime - ptime)
    ptime = cTime
    # cv2.putText(img, str(int(fps)), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (125, 128, 225), 3)

    # display
    image = cv2.flip(img, flipCode=1)
    cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (125, 128, 5), 3)
    cv2.imshow("Video", image)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
