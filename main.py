import numpy as np
import cv2 as cv

# 1080 x 1920
H = 1080
W = 1920
SPEAR = "data/input/2021-01-16 21-38-22.mkv"
POLEARM = "data/input/2021-01-16 21-41-58.mkv"
cap = cv.VideoCapture(POLEARM)

while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.rectangle(
        gray,
        ((W // 2) - 250, (H // 2 + 50)),
        ((W // 2) + 50, (H // 2 + 350)),
        (0, 255, 0),
        3,
    )
    cv.imshow("frame", gray)
    if cv.waitKey(1) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()