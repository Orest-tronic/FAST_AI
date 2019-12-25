import cv2
import numpy as np
import time
import urllib
# import imutils
from pathlib import Path
from fastai.vision.data import ImageDataBunch, get_transforms, imagenet_stats
from fastai.vision.learner import cnn_learner
from fastai.vision import models
from fastai.vision.image import pil2tensor, Image
import csv


#URL = "http://192.168.43.1:8080/shot.jpg"
def read_labels(path: str):
    with open(path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        labels_map = {}
        row_index = 0
        for row in csv_reader:
            if row_index == 0:
                 #print(1)
                print(f'Column names are {", ".join(row)}')
            else:
                print(f'\t{row["id"]} => {row["name"]} ')
                labels_map[row["id"]] = row["name"]
            row_index += 1
        print(f'Processed {row_index} lines.')
    return labels_map


LABEL_MAP = read_labels('Training/labels.csv')

#imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# data = (
#         ImageList.from_csv("Training", csv_name="labels.csv")
#         .no_split()
#         .label_from_df(label_delim=" ")
#         .transform(None, size=128)
#         .databunch(no_check=True)
#         .normalize(imagenet_stats)
#     )
path = Path("Training")
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2, ds_tfms=get_transforms(), num_workers=4,
                                  size=224).normalize(imagenet_stats)
# Loading our model
learn = cnn_learner(data, models.resnet34, pretrained=False)
learn.load("stage-2")
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
kernel = np.ones((3, 3), np.uint8)
Traffic = cv2.CascadeClassifier('Final.xml')
# img_source = np.array(bytearray(urllib.request.urlopen(URL).read()), dtype=np.uint8)
# img = cv2.imdecode(img_source,-1)

# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter("enreg3.mkv", fourcc, 9.0, (640, 480))
while True:

    # img_source = np.array(bytearray(urllib.request.urlopen(URL).read()), dtype=np.uint8)
    # img = cv2.imdecode(img_source,-1)
    # cv2.imshow('CAMDIRECT VOCK', img)
    ret, img = cap.read()
    #cv2.imshow('Originale', original)
    #img = original
    if type(img) == type(None):
        break
    #kernel = np.ones((3, 3), np.uint8)
    # img = cv2.erode(img, kernel, iterations=1)
    # img = cv2.dilate(img, kernel, iterations=1)
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    gray = cv2.erode(gray, kernel, iterations=1)
    gray = cv2.dilate(gray, kernel, iterations=2)
    #cv2.imshow('test_gray', gray)
    Traffic_sign = Traffic.detectMultiScale(gray, 1.1, 5)  # 1.05

    for coords in Traffic_sign:
        X, Y, w, h = coords

        ## Finding frame size
        H, W, _ = img.shape

        ## Computing larger face co-ordinates
        # X_1, X_2 = (max(0, X - int(w * 0.35)), min(X + int(1.35 * w), W))
        # Y_1, Y_2 = (max(0, Y - int(0.35 * h)), min(Y + int(1.35 * h), H))
        X_1, X_2 = (max(0, X - int(w)), min(X + int(w), W))
        Y_1, Y_2 = (max(0, Y - int(h)), min(Y + int(h), H))
        ## Cropping face and changing BGR To RGB
        img_cp = img[Y_1:Y_2, X_1:X_2].copy()
        img_cp1 = cv2.cvtColor(img_cp, cv2.COLOR_BGR2RGB)

        ## Prediction of facial featues
        prediction = str(
            learn.predict(Image(pil2tensor(img_cp1, np.float32).div_(255)))[0]
        ).split(";")
        # label = (
        #   " ".join(prediction)
        #  if "Male" in prediction
        #  else "Female " + " ".join(prediction)
        # )
        label = (
            " ".join(prediction)
            # if "No_Beard" in prediction
            # else "Beard " + " ".join(prediction)
        )
        img = cv2.rectangle(img, (X, Y), (X + w, Y + h), (150, 0, 100), 2)

        ## Drawing facial attributes identified
        label_list = label.split(" ")
        for idx in range(1, len(label_list) + 1):
            cv2.putText(
                img,
                LABEL_MAP[label_list[idx - 1]],
                (X, Y - 14 * idx),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (80, 100, 50),
                2,
            )

        # area = (x, y, x+w, y+h)
    # if area != []:
    # cv2.imwrite("C:/Users/paloma/PycharmProjects/test_image/frame.jpg", img)
    # test = Image.open("C:/Users/paloma/PycharmProjects/test_image/frame.jpg")
    # cropped = test.crop(area)
    # pil_image = PIL.Image.open('image.jpg')
    # cropped = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
    # cv2.imshow('cropped', cropped)

    #out.write(img)
    cv2.imshow('video', img)  # cv2.resize(img, (200, 400)))
    # cv2.imshow('gray', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        #out.release()
        break
cv2.destroyAllWindows()
