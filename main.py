import pandas as pd
import cv2
import numpy as np
from shape_detection import Shape_detection as Sd


BASE_DIR = './test/test/'


def metod_Oocu(img):
    min = img[0, 0][2]
    max = img[0, 0][2]

    for i in range(len(img)):
        for j in range(len(img[i])):
            v = img[i][j][2]
            if v < min:
                min = v
            if v > max:
                max = v

    histSize = max - min + 1
    hist = []
    for i in range(histSize):
        hist.append(0)

    for i in range(len(img)):
        for j in range(len(img[i])):
            v = img[i][j][2]
            hist[v-min] += 1

    m = 0
    n = 0

    for t in range(histSize):
        m += t * hist[t]
        n += hist[t]

    maxSigma = -1
    T = 0
    alpha1 = 0
    beta1 = 0

    for t in range(histSize-1):
        alpha1 += t * hist[t]
        beta1 += hist[t]

        w1 = beta1 / n
        a = alpha1 / beta1 - (m - alpha1) / (n - beta1)

        sigma = w1 * (1 - w1) * a * a

        if sigma > maxSigma:
            maxSigma = sigma
            T = t

    T += min
    return T


def handle_image(image_name):
    img = cv2.imread(BASE_DIR + image_name)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # # low = (276, 32, 82)
    # # high = (255, 255, 255)
    #
    # low = (0,0,0)
    # high = (240,240,240)
    # mask = cv2.inRange(img, low, high)
    # img = cv2.bitwise_and(img, img, mask=mask)

    # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    T = metod_Oocu(img_hsv)

    lower_white = np.array([0, 0, 0])
    upper_white = np.array([255, 255, T])


    mask = cv2.inRange(img_hsv, lower_white, upper_white)

    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, None)
    mask = cv2.blur(mask, (20, 20))
    mask = cv2.bitwise_not(mask)

    target = cv2.bitwise_and(img, img, mask=mask)
    # cv2.imshow("shapes", target)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    F = False
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) > 2:
            shape = Sd(approx)
            if shape.draw(img):
                rect = cv2.minAreaRect(cnt)  # пытаемся вписать прямоугольник
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                area = int(rect[1][0] * rect[1][1])
                if area > 500:
                    if (abs(box[0][0]-box[1][0]) > 1.5*abs(box[0][1]-box[3][1])) or (abs(box[0][1]-box[3][1]) > 1.5*abs(box[0][0]-box[1][0])):
                        cv2.drawContours(img, [box], 0, (255, 0, 0), 0)
                        F = True
                        return 1
    return 0


def main():
    df = pd.read_csv('./sample_submission.csv')

    for index, row in df.iterrows():
        df.at[index, 'IdCls'] = handle_image(row[0])

    df.to_csv('./sample_submission.csv', index=False)


if __name__ == "__main__":
    main()
