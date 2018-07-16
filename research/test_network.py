# USAGE
# python classify.py --model pokedex.model --labelbin lb.pickle --image examples/charmander_counter.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import pickle
import cv2
import os

def identifyCharacter():
    # load the image
    image = cv2.imread("segment.jpg")
    output = image.copy()

    # pre-process the image for classification
    image = cv2.resize(image, (75, 75))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # load the trained convolutional neural network and the label
    # binarizer
    print("[INFO] loading network...")
    model = load_model("characterCls.model")
    lb = pickle.loads(open("lb.pickle", "rb").read())

    # classify the input image
    print("[INFO] classifying image...")
    proba = model.predict(image)[0]
    #print(proba)
    idx = np.argmax(proba)
    #print(idx)
    label = lb.classes_[idx]
    #print(label)
    # we'll mark our prediction as "correct" of the input image filename
    # contains the predicted label text (obviously this makes the
    # assumption that you have named your testing image files this way)
    filename = "segment.jpg"["segment.jpg".rfind(os.path.sep)+1:]
    print(filename)
    correct = "correct" if filename.rfind(label) != -1 else "incorrect"

    # build the label and draw the label on the image
    clss = label
    if correct == "correct":
        label = "{}: {:.2f}% ".format(label, proba[idx] * 100)
        output = imutils.resize(output, width=400)
        cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        # show the output image
        print("[INFO] {}".format(label))
        cv2.imshow("Output", output)

    else:
        print("[INFO] {}: {:.2f}% ({})".format(label, proba[idx] * 100, correct))
        idntify = "cannot identify"
        label = "{}".format(idntify)
        output = imutils.resize(output, width=400)
        cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        # show the output image

        cv2.imshow("Output", output)

    return clss


