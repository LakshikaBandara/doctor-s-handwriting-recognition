import cv2

class preProcessing:
    def basicProcess(img):
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # get grayscale image
        imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)  # blur

        # filter image from grayscale to black and white
        imgThresh = cv2.adaptiveThreshold(imgBlurred,  # input image
                                          255,  # make pixels that pass the threshold full white
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          # use gaussian rather than mean, seems to give better results
                                          cv2.THRESH_BINARY_INV,
                                          # invert so foreground will be white, background will be black
                                          11,  # size of a pixel neighborhood used to calculate threshold value
                                          2)  # constant subtracted from the mean or weighted mean

        return imgThresh

    def normalChar(img):
        dilation = cv2.dilate(img, (3, 3), iterations=15)
        return dilation

    def cursiveChar(img):
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, (5, 5))
        return opening