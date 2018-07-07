# importing libries
import sys
import numpy as np
import cv2
import os
from test_network import identifyCharacter
from preProcessing import preProcessing

MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30


def main():
    trainImage = cv2.imread("testData/c.jpg")  # read in training numbers image

    if trainImage is None:  # if image was not read successfully
        print("error: image not read from file \n\n")  # print error message to std out
        os.system("pause")  # pause so user can see error message
        return  # and exit function (which exits program)
    # end ifll
    cv2.imshow("Image", trainImage)

    imgThresh = preProcessing.basicProcess(trainImage)

    key = cv2.waitKey(0)
    preProcessedImg = imgThresh.copy()

    if key == 60: #if cursive character press <
        preProcessedImg = preProcessing.cursiveChar(imgThresh)  # make a copy of the thresh image, this in necessary b/c findContours modifies the image

    elif key == 62: #if normal character press >

        preProcessedImg = preProcessing.normalChar(imgThresh)  # make a copy of the thresh image, this in necessary b/c findContours modifies the image

    imgContours, npaContours, npaHierarchy = cv2.findContours(preProcessedImg,
                                                              # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                              cv2.RETR_EXTERNAL,  # retrieve the outermost contours only
                                                              cv2.CHAIN_APPROX_SIMPLE)  # compress horizontal, vertical, and diagonal segments and leave only their end points

    # declare empty numpy array, we will use this to write to file later
    # zero rows, enough cols to hold all image data
    npaFlattenedImages = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

    intClassifications = []  # declare empty classifications list, this will be our list of how we are classifying our chars from user input, we will write to file at the end

    # possible chars we are interested in are digits 0 through 9, put these in list intValidChars

    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('a'), ord('b'), ord('c'), ord('d'), ord('e'), ord('f'), ord('g'), ord('h'), ord('i'), ord('j'),
                     ord('k'), ord('l'), ord('m'), ord('n'), ord('o'), ord('p'), ord('q'), ord('r'), ord('s'), ord('t'),
                     ord('u'), ord('v'), ord('w'), ord('x'), ord('y'), ord('z'),ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]

    for npaContour in npaContours:  # for each contour
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:  # if contour is big enough to consider
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)  # get and break out bounding rect

            # draw rectangle around each contour as we ask user for input
            cv2.rectangle(trainImage,  # draw rectangle on original training image
                          (intX, intY),  # upper left corner
                          (intX + intW, intY + intH),  # lower right corner
                          (0, 0, 255),  # red
                          2)  # thickness

            imgROI = imgThresh[intY:intY + intH, intX:intX + intW]  # crop char out of threshold image
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH,
                                                RESIZED_IMAGE_HEIGHT))  # resize image, this will be more consistent for recognition and storage

            cv2.imwrite("testData/segment.jpg", imgROIResized)
            cv2.imshow("training image.png",
                       trainImage)  # show training numbers image, this will now have red rectangles drawn on it
            charLable = identifyCharacter()
            intChar = cv2.waitKey(0)
            print(intChar)

            if intChar == 27:  # if esc key was pressed
                sys.exit()  # exit program
            elif intChar == 13:  # else if the char is in the list of chars we are looking for . . .

                intClassifications.append(
                    ord(charLable))  # append classification char to integer list of chars (we will convert to float later before writing to file)

                npaFlattenedImage = imgROIResized.reshape((1,
                                                           RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # flatten image to 1d numpy array so we can write to file later
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage,
                                               0)  # add current flattened impage numpy array to list of flattened image numpy arrays
            elif intChar in intValidChars:  # else if the char is in the list of chars we are looking for . . .

                intClassifications.append(
                    intChar)  # append classification char to integer list of chars (we will convert to float later before writing to file)

                npaFlattenedImage = imgROIResized.reshape((1,
                                                           RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # flatten image to 1d numpy array so we can write to file later
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage,
                                               0)
            # end if
        # end if
    # end for

    fltClassifications = np.array(intClassifications,np.float32)  # convert classifications list of ints to numpy array of floats

    npaClassifications = fltClassifications.reshape(
    (fltClassifications.size, 1))  # flatten numpy array of floats to 1d so we can write to file later

    clasfile = open('classifications.txt', 'a')
    np.savetxt(clasfile, npaClassifications)
    clasfile.close()

    fatFile = open('flattened_images.txt', 'a')
    np.savetxt(fatFile, npaFlattenedImages)
    fatFile.close()

    print("\n\ntraining complete !!\n")
    cv2.destroyAllWindows()  # remove windows from memory

    return


###################################################################################################
if __name__ == "__main__":
    main()
# end if
