import cv2
import numpy as np
#import time
#import urllib
from threading import Thread, RLock
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
                print(1)
            else:
                labels_map[row["id"]] = row["name"]
            row_index += 1
    return labels_map


count = 0
LABEL_MAP = read_labels('Training/labels.csv')
verrou = RLock()


class Position(Thread):

    def __init__(self, coords, imgs):
        Thread.__init__(self)
        self.coord = coords
        self.image = imgs

    def run(self):
        with verrou:
            H, W, C = self.image.shape
            X, Y, w, h = self.coord
            X_1, X_2 = (max(0, X - int(w)), min(X + int(w), W))
            Y_1, Y_2 = (max(0, Y - int(h)), min(Y + int(h), H))
            img_cp = self.image[Y_1:Y_2, X_1:X_2].copy()
            img_cp1 = cv2.cvtColor(img_cp, cv2.COLOR_BGR2RGB)
            prediction = str(
                learn.predict(Image(pil2tensor(img_cp1, np.float32).div_(255)))[0]
            ).split(";")
            label = (
                " ".join(prediction)
                # if "No_Beard" in prediction
                # else "Beard " + " ".join(prediction)
            )
            label_list = label.split(" ")
            self.image = cv2.rectangle(self.image, (X, Y), (X + w, Y + h), (150, 0, 100), 2)
            for idx in range(1, len(label_list) + 1):
                cv2.putText(
                    self.image,
                    LABEL_MAP[label_list[idx - 1]],
                    (X, Y - 14 * idx),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (80, 100, 50),
                    2,
                )
            print("Label :", label)


path = Path("Training")
tfm = get_transforms(do_flip=True, max_rotate=35.0, max_zoom=0.6, max_lighting=0.3, max_warp=0.2, p_affine=0.75, p_lighting=0.75)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2, ds_tfms=tfm, num_workers=4, size=224).normalize(imagenet_stats)
# Loading our model
learn = cnn_learner(data, models.resnet50, pretrained=False)
learn.load("stage-3")
cap = cv2.VideoCapture('testet.mp4')
Traffic = cv2.CascadeClassifier('second_2_5.xml')
while True:
    ret, img = cap.read()
    if type(img) == type(None):
        break
    ret = count % 5
    if ret == 0:
        H, W, C = img.shape
        gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        Traffic_sign = Traffic.detectMultiScale(gray, scaleFactor=2, minNeighbors=5, minSize=(90, 90), maxSize=(120, 120))  # 1.05
        if len(Traffic_sign) < 1:
            print("NOTHING FOUND")
        elif len(Traffic_sign) < 2:
            print("Found 1")
            X, Y, w, h = Traffic_sign[0]
            X_1, X_2 = (max(0, X - int(w)), min(X + int(w), W))
            Y_1, Y_2 = (max(0, Y - int(h)), min(Y + int(h), H))
            img_cp = img[Y_1:Y_2, X_1:X_2].copy()
            img_cp1 = cv2.cvtColor(img_cp, cv2.COLOR_BGR2RGB)
            prediction = str(
                learn.predict(Image(pil2tensor(img_cp1, np.float32).div_(255)))[0]
            ).split(";")
            img = cv2.rectangle(img, (X, Y), (X + w, Y + h), (150, 0, 100), 2)
            label = (
                " ".join(prediction)
                # if "No_Beard" in prediction
                # else "Beard " + " ".join(prediction)
                )
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
            print("Label :", label)
        elif len(Traffic_sign) < 3:
            print("Found 2")
            thread_1 = Position(Traffic_sign[0], img)
            thread_2 = Position(Traffic_sign[1], img)
            thread_1.start()
            thread_2.start()
            thread_1.join()
            thread_2.join()
        else:
            print("Found Multiple")
            thread_1 = Position(Traffic_sign[0], img)
            thread_2 = Position(Traffic_sign[1], img)
            thread_3 = Position(Traffic_sign[1], img)
            thread_1.start()
            thread_2.start()
            thread_3.start()
            thread_1.join()
            thread_2.join()
            thread_3.join()
    cv2.imshow('video', img)
    count = count + 1
    print("Count= ", count)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
