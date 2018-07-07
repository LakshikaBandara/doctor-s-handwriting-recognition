import cv2
import imutils
def saveImg(img):

    blur = cv2.blur(img, (5, 5)) # average blurred
    blur = cv2.GaussianBlur(img, (5, 5), 0) # gaussian filter
    median = cv2.medianBlur(img, 5) # median filter
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, (3,3)) #opening
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, (3,3)) #closing
    rotated = imutils.rotate(img, 15) # rotate 15 degree

    return