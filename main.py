import cv2
import numpy as np

from numpy import ndarray, dtype, float64
from typing import *

net: cv2.dnn.Net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classlist: list[str] = [i.strip() for i in open('coco.names', 'r').readlines()] # Object class types recognizable
layer_names: Sequence[str] = net.getLayerNames()
output_layers: list[str] = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colorlist: ndarray[Any, dtype[float64]] = np.random.uniform(0, 255, size=(len(classlist), 3))

# Image Loading
img: cv2.typing.MatLike = cv2.imread("sample.jpg")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
(height, width, channels): int = img.shape

# Detect Objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# 정보를 화면에 표시
class_ids: list[np.intp] = []
confidences: list[float] = []
boxes: list[list[int, int, int, int]] = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5: # If the confidence over half ( Adjustable )
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Position
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

indexes: Sequence[int] = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font: int = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classlist[class_ids[i]])
        color = colorlist[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
 