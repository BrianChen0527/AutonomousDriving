import cv2

cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()

    # original colors
    colorful = cv2.cvtColor(frame, cv2.IMREAD_COLOR)

    cv2.imshow('frame', colorful)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()