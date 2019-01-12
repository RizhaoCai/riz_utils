import cv2
from skimage import color


def colorspace_transform(im_src, dst_space, src_space="BGR"):
    """
        target_space: HSV, BGR, YCBCR
    """
    if src_space == "BGR":
        if dst_space == "HSV":
            im_dst = cv2.cvtColor(im_src,cv2.COLOR_BGR2HSV)
        elif dst_space == "YCrCb":
            im_dst = cv2.cvtColor(im_src,cv2.COLOR_BGR2YCrCb)
        elif dst_space == "gray":
            im_dst = cv2.cvtColor(im_src,cv2.COLOR_BGR2GRAY)

    return im_dst

def batch_rgb2hsv():
    pass


    