import cv2
import numpy as np
import utils
import cairo
bg = cv2.imread('misc/scoop-4-1.jpg')
bg_ = []
for ymin in [5, 147, 298, 450, 601]:
    bg_.append(cv2.resize(bg[ymin:ymin + 86, 1:201], (226, 70)))
    bg_.append(cv2.resize(bg[ymin:ymin + 86, 207:405], (226, 70)))


def random_bg():
    return bg_[np.random.randint(0, len(bg_))]


if __name__ == '__main__':

    bgr = utils.paint_text('ขก 1242', 112, np.random.randint(77), aug=False, test=True, useabg=False, randfont=True)
    cv2.imshow('bgr', bgr)
    cv2.waitKey()
    cv2.destroyAllWindows()
