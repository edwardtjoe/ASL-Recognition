import numpy as np
import os
import cv2

image_x, image_y = 224, 224

cap = cv2.VideoCapture(0)
# fbag = cv2.createBackgroundSubtractorMOG2()
# fbag = cv2.createBackgroundSubtractorKNN()


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


def main(g_id):
    total_pics = 2000
    cap = cv2.VideoCapture(0)
    x, y, w, h = 100, 50, 350, 350

    create_folder("gestures/train/" + str(g_id))
    pic_no = 0
    flag_start_capturing = False
    frames = 0

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask2 = cv2.inRange(hsv, np.array([2, 50, 60]), np.array([25, 150, 255]))
        res = cv2.bitwise_and(frame, frame, mask=mask2)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        median = cv2.GaussianBlur(gray, (5, 5), 0)

        kernel = np.ones((5, 5), np.uint8)
        # erosion = cv2.erode(median, kernel_square, iterations=1)
        # dilation = cv2.dilate(erosion, kernel_square, iterations=1)
        morph_1 = cv2.morphologyEx(median, cv2.MORPH_GRADIENT, kernel, iterations=1)  # OPEN = ERODE + DILATE
        morph_2 = cv2.morphologyEx(morph_1, cv2.MORPH_ELLIPSE, kernel, iterations=1)  # GRADIENT = Take outlines

        ret, thresh = cv2.threshold(morph_1, 30, 255, cv2.THRESH_BINARY)
        thresh = thresh[y:y + h, x:x + w]

        ret_2, thresh_2 = cv2.threshold(morph_2, 30, 255, cv2.THRESH_BINARY)
        thresh_2 = thresh
        # thresh_2 = thresh_2[y:y + h, x+2*w:x + 3*w]

        contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        # print(f'contours:{contours}')

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            # contour_2 = max(contours_2, key=cv2.contourArea)
            if cv2.contourArea(contour) > 10000 and frames > 50:
                x1, y1, w1, h1 = cv2.boundingRect(contour)

                pic_no += 1
                save_img = thresh[y1:y1 + h1, x1:x1 + w1]
                if w1 > h1:
                    save_img = cv2.copyMakeBorder(save_img, int((w1 - h1) / 2), int((w1 - h1) / 2), 0, 0,
                                                  cv2.BORDER_CONSTANT, (0, 0, 0))

                else:
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1 - w1) / 2), int((h1 - w1) / 2),
                                                  cv2.BORDER_CONSTANT, (0, 0, 0))

                save_img = cv2.resize(save_img, (image_x, image_y))

                cv2.putText(frame, "TAKING PICTURE", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255))
                cv2.imwrite("gestures/train/" + str(g_id) + "/" + str(pic_no) + ".jpg", save_img)
            else:
                cv2.putText(frame, "COME CLOSER", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255))
        else:
            cv2.putText(frame, "IDLE", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255))

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.rectangle(frame, (x+2*w, y), (x + 3*w, y + h), (0, 0, 255), 2)                # NOTE: Uncomment for two rectangles

        cv2.putText(frame, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        cv2.imshow("Capturing gesture", frame)
        cv2.imshow("thresh", thresh)

        cv2.imshow("thresh_2", thresh_2)

        keypress = cv2.waitKey(1)
        if keypress == ord('c'):
            if not flag_start_capturing:
                flag_start_capturing = True
            else:
                flag_start_capturing = False
                frames = 0
        if flag_start_capturing:
            frames += 1
        if keypress == ord('q'):
            break
        if pic_no == total_pics:
            break


g_id = input("Enter gesture number: ")
main(g_id)
