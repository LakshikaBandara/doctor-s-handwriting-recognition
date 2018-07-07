from tkinter import *
import numpy as np
import cv2
import os
import operator
from tkinter import scrolledtext
from tkinter import filedialog
import tkinter
from tkinter import messagebox
from PIL import Image, ImageTk
from characterRecognition.testing import train
textpredicted = []
filePath = []
filePathTrain = []

def clicked():
    MIN_CONTOUR_AREA = 100

    RESIZED_IMAGE_WIDTH = 20
    RESIZED_IMAGE_HEIGHT = 30

    ###################################################################################################
    class ContourWithData():
        # member variables ############################################################################
        npaContour = None  # contour
        boundingRect = None  # bounding rect for contour
        intRectX = 0  # bounding rect top left corner x location
        intRectY = 0  # bounding rect top left corner y location
        intRectWidth = 0  # bounding rect width
        intRectHeight = 0  # bounding rect height
        fltArea = 0.0  # area of contour

        def calculateRectTopLeftPointAndWidthAndHeight(self):  # calculate bounding rect info
            [intX, intY, intWidth, intHeight] = self.boundingRect
            self.intRectX = intX
            self.intRectY = intY
            self.intRectWidth = intWidth
            self.intRectHeight = intHeight

        def checkIfContourIsValid(self):  # this is oversimplified, for a production grade program
            if self.fltArea < MIN_CONTOUR_AREA: return False  # much better validity checking would be necessary
            return True

    ###################################################################################################
    def main():
        allContoursWithData = []  # declare empty lists,
        validContoursWithData = []  # we will fill these shortly

        try:
            npaClassifications = np.loadtxt("classifications.txt", np.float32)  # read in training classifications
        except:
            print("error, unable to open classifications.txt, exiting program\n")
            os.system("pause")
            return
        # end try

        try:
            npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)  # read in training images
        except:
            print("error, unable to open flattened_images.txt, exiting program\n")
            os.system("pause")
            return
        # end try

        npaClassifications = npaClassifications.reshape(
            (npaClassifications.size, 1))  # reshape numpy array to 1d, necessary to pass to call to train

        kNearest = cv2.ml.KNearest_create()  # instantiate KNN object

        kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
        f = str(filePath)
        print(f)
        imgTestingNumbers = cv2.imread(filePath.pop(0))  # read in testing numbers image

        if imgTestingNumbers is None:  # if image was not read successfully
            print("error: image not read from file \n\n")  # print error message to std out
            os.system("pause")  # pause so user can see error message
            return  # and exit function (which exits program)
        # end if

        imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)  # get grayscale image
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

        imgThreshCopy = imgThresh.copy()  # make a copy of the thresh image, this in necessary b/c findContours modifies the image

        imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,
                                                                  # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                                  cv2.RETR_EXTERNAL,
                                                                  # retrieve the outermost contours only
                                                                  cv2.CHAIN_APPROX_SIMPLE)  # compress horizontal, vertical, and diagonal segments and leave only their end points

        for npaContour in npaContours:  # for each contour
            contourWithData = ContourWithData()  # instantiate a contour with data object
            contourWithData.npaContour = npaContour  # assign contour to contour with data
            contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)  # get the bounding rect
            contourWithData.calculateRectTopLeftPointAndWidthAndHeight()  # get bounding rect info
            contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)  # calculate the contour area
            allContoursWithData.append(
                contourWithData)  # add contour with data object to list of all contours with data
        # end for

        for contourWithData in allContoursWithData:  # for all contours
            if contourWithData.checkIfContourIsValid():  # check if valid
                validContoursWithData.append(contourWithData)  # if so, append to valid contour list
            # end if
        # end for

        validContoursWithData.sort(key=operator.attrgetter("intRectX"))  # sort contours from left to right

        strFinalString = ""  # declare final string, this will have the final number sequence by the end of the program

        for contourWithData in validContoursWithData:  # for each contour
            # draw a green rect around the current char
            cv2.rectangle(imgTestingNumbers,  # draw rectangle on original testing image
                          (contourWithData.intRectX, contourWithData.intRectY),  # upper left corner
                          (contourWithData.intRectX + contourWithData.intRectWidth,
                           contourWithData.intRectY + contourWithData.intRectHeight),  # lower right corner
                          (0, 255, 0),  # green
                          2)  # thickness

            imgROI = imgThresh[contourWithData.intRectY: contourWithData.intRectY + contourWithData.intRectHeight,
                     # crop char out of threshold image
                     contourWithData.intRectX: contourWithData.intRectX + contourWithData.intRectWidth]

            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH,
                                                RESIZED_IMAGE_HEIGHT))  # resize image, this will be more consistent for recognition and storage

            npaROIResized = imgROIResized.reshape(
                (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # flatten image into 1d numpy array

            npaROIResized = np.float32(npaROIResized)  # convert from 1d numpy array of ints to 1d numpy array of floats

            retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,
                                                                         k=1)  # call KNN function find_nearest

            strCurrentChar = str(chr(int(npaResults[0][0])))  # get character from results

            strFinalString = strFinalString + strCurrentChar  # append current char to full string
        # end for

        print("\n" + strFinalString + "\n")  # show the full string
        textpredicted.append(strFinalString)
        print(textpredicted)
        txt.insert(INSERT, textpredicted)
        cv2.imshow("imgTestingNumbers",
                   imgTestingNumbers)  # show input image with green boxes drawn around found digits
        cv2.waitKey(0)  # wait for user key press

        cv2.destroyAllWindows()  # remove windows from memory

        return

    ###################################################################################################
    if __name__ == "__main__":
        main()
    # end if


def trained():
    train(filePathTrain.pop(0))
    #cv2.imread(filePathTrain.pop(0))
    #  read in training numbers image




def upload():
    file = filedialog.askopenfile(mode='rb', title='Choose a file')
    if file != None:
        data = file.read()
        file.close()
        print("I got %d bytes from this file." % len(data))
        filePath.append(file.name)
        # print(filePath)]

        im = Image.open(file.name)
        img = im.resize((200, 200),Image.ANTIALIAS)
        ph = ImageTk.PhotoImage(img)

        #label = Label(window, image=ph)
        #label.image = ph  # need to keep the reference of your image to avoid garbage collection

        lbl2 = Label(window, text="Train", image=ph)
        lbl2.image = ph

        lbl2.place(x=600,y=50,height=200,width=200)

        #lbl2.grid(column=10, row=4)

def uploadTrain():
    file = filedialog.askopenfile(mode='rb', title='Choose a file')
    if file != None:
        data = file.read()
        file.close()
        print("I got %d bytes from this file." % len(data))
        filePathTrain.append(file.name)




def saveDoc():
    # np.save("asd.doc", textpredicted.pop(0))
    f = filedialog.asksaveasfile(mode='w', defaultextension=".doc")
    if f is None:  # asksaveasfile return `None` if dialog closed with "cancel".
        return
    captured = str(txt.get("1.0", tkinter.END))
    text2save = str(captured)  # starts from `1.0`, not `0.0`
    f.write(text2save)
    f.close()
    messagebox.showinfo("Title", "File Saved Suceesfully")


######################################################Jython UI#######################################
window = Tk()

window.title("Image to Text")

window.geometry('800x400')

lbl = Label(window, text="Converted Text Below")
#lbl.grid(column=2, row=2)
lbl.place(x=180,y=20)
#lbl.pack()
lblUploaded = Label(window, text="Uploaded Image Below")
lblUploaded.place(x=600,y=20)
txt = scrolledtext.ScrolledText(window, width=40, height=10)
#txt.grid(column=2, row=4)

#txt.pack()
txt.place(x=100,y=40)

photo=PhotoImage(file="icons/file.png")
b = Button(window,image=photo, text="Upload", command=uploadTrain, height=25, width=55, compound=RIGHT)
b.place(x = 360, y = 220)


photo2=PhotoImage(file="icons/idea.png")
btn = Button(window,image=photo2, text="Convert", command=clicked,height=25, width=55, compound=RIGHT)
#btn.grid(column=4, row=2)
btn.place(x = 200, y = 220)


photo3=PhotoImage(file="icons/tool.png")
btn2 = Button(window,image=photo2, text="Train", command=trained,height=25, width=55, compound=RIGHT)
btn2.place(x = 280, y = 220)
#btn2 = Button(window, text="Train", command=trained, height=25, width=55)
#btn2.grid(column=5, row=2)
#btn2.place(x =300, y = 400)
#btnUpload = Button(window, text="Upload", command=upload, height=25, width=55)
#btnUpload.grid(column=6, row=2)
#btnUpload = Button(window, text="Save", command=saveDoc)
#btnUpload.grid(column=7, row=2)

menubar = Menu(window)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Open",command=upload)
filemenu.add_command(label="Save",command=saveDoc)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=window.quit)
menubar.add_cascade(label="File", menu=filemenu)

helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="Help Index")
helpmenu.add_command(label="About...")
menubar.add_cascade(label="Help", menu=helpmenu)


#button = Button(window, text="Click me!")
#img = PhotoImage(file="C:/Users/kasun_000/Downloads/file.png") # make sure to add "/" not "\"
#button.config(image=img,command = upload,text="Upload")

#button.pack() # Displaying the button


window.config(menu=menubar)
window.config(background='gray80')
window.mainloop()
