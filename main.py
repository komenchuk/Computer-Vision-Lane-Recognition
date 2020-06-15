# Импортируем нужные для даной задачи библиотеки
import cv2 as cv
import numpy as np
from ComputerVisionLaneRecognition import lanes

video = cv.VideoCapture("road.mp4")

if not video.isOpened():
    print('Error while opening the video')

cv.waitKey(1)

while video.isOpened():
    _, frame = video.read()
    copy_img = np.copy(frame)

    try:
        frame = lanes.canny(frame)
        frame = lanes.mask(frame)
        lines = cv.HoughLinesP(frame, 2, np.pi / 180, 100, np.array([()]), minLineLength=20, maxLineGap=5)
        averaged_lines = lanes.average_slope_intercept(frame, lines)
        line_image = lanes.display_lines(copy_img, averaged_lines)

        combo = cv.addWeighted(copy_img, 0.8, line_image, 0.5, 1)
        cv.imshow("Video", combo)
    except:
        pass

    if cv.waitKey(1) & 0xFF == ord('q'):
        video.release()
        cv.destroyAllWindows()

video.release()
cv.destroyAllWindows()
