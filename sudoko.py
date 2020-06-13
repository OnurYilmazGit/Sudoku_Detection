import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt



def sudoko_extractor(path):

    for f1 in glob.glob(path):
    
        img = cv2.imread(f1)
    
        # Add weighted function used to increase contrast of image.
        img = cv2.addWeighted(img, 1.25, np.zeros(img.shape, img.dtype), 0, -30)
    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # resized to makes extracting the grid lines easier.
        gray = cv2.resize(gray, None, fx=0.8, fy=0.8)
    
        mask = np.zeros((gray.shape), np.uint8)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    
        close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel1)
    
        div = np.float32(gray) / (close)
    
        res = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))
    
        #Adaptive threshold using 15 nearest neighbour pixels
        thresh = cv2.adaptiveThreshold(res, 255, 1, 1, 15, 2)
        #findContours algorithm detects shapes in the image.
        contour, hier = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
        # Finding the biggest rectange with using contour feature.
        max_area = 0
        best_cnt = None
        for cnt in contour:
            area = cv2.contourArea(cnt)
            if area > 1000:
                if area > max_area:
                    max_area = area
                    best_cnt = cnt
        # Draw all of the contours on the image in 2px lines.
        cv2.drawContours(mask, [best_cnt], 0, 255, -1)
        cv2.drawContours(mask, [best_cnt], 0, 0, 2)
    
        res = cv2.bitwise_and(res, mask)
    
        # obtain sharper image
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        res = cv2.filter2D(res, -1, kernel)
    
        # Edge detection using specific parameters
        edges = cv2.Canny(gray, 48, 280, apertureSize=3)
    
        # Used the Hough transform to get lines in this image. It returns lines in
        # mathematical terms.
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 145)
        # If lines has no type error, it'll calculate on try.
        try:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(res, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # If edges could not detected well, it will calculate again using string
        # parameters.
        except BaseException:
            edges = cv2.Canny(gray, 100, 200, apertureSize=5)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 165)
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(res, (x1, y1), (x2, y2), (0, 0, 255), 2)
            pass
        # MediaBlur used to recude noise of image.
        res = cv2.medianBlur(res, 3)
        # Image showing function.
        cv2.imshow('edges', res)
    
        k = cv2.waitKey(0)
        if k == 27:
            break
        cv2.destroyAllWindows()

if __name__ == '__main__':
    path = '/Users/onur_yilmaz/Desktop/sudoku_dataset-master/images/*.jpg'
    sudoko_extractor(path)