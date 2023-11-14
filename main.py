import cv2
import numpy as np

from numpy import ndarray, dtype, float32, float64, intp
from typing import *
from cv2.typing import *

CONFIDENCE_CRITICAL_NUMBER = 0.5

# 사전 학습된 YOLO 모델과 cfg 파일을 로드한다.
# https://pjreddie.com/darknet/yolo/ < 해당 사이트에서 원하는 모델을 다운받을 수 있다.
net: cv2.dnn.Net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# 인식 가능한 객체 클래스 이름을 로드한다.
classList: list[str] = [i.strip() for i in open('coco.names', 'r').readlines()]

# YOLO 네트워크 레이어 이름 가져오기
# 신경망의 모든 레이어의 이름을 담은 문자열 목록을 반환한다.
# 이 목록은 입력층, 출력층, 은닉층 등을 모두 포함한다.
layerNameList: Sequence[str] = net.getLayerNames()

# 출력 레이어의 이름 가져오기
# getUnconnectedOutLayers() 메소드는 연결되지 않은 출력 레이어의 인덱스를 반환한다.
# 네트워크를 거친 결과를 출력하기 위해 출력 레이어들을 모으는 작업이다.
outputLayerList: list[str] = [layerNameList[i - 1] for i in net.getUnconnectedOutLayers()]

# 각각의 객체 클래스 상자에 대한 랜덤한 색상을 생성하여 저장한다.
colorList: ndarray[Any, dtype[float64]] = np.random.uniform(0, 255, size=(len(classList), 3))

# 샘플 이미지 로드 및 크기 조정
img: MatLike = cv2.resize(cv2.imread("sample.jpg"), None, fx=0.4, fy=0.4)

# 이미지 크기 정보를 저장한다.
height, width, channels = img.shape 


# 앞서 불러온 img 파일을 네트워크에 사용 가능한 형식인 blob으로 변환시킨다.
# 중간에 들어간 scalefactor 값은 이미지의 픽셀 값 범위 스케일링에 쓰이는 계수이다. 이미지의 모든 픽셀을 255배 축소시킨다.
# 이 작업을 하는 이유는 네트워크가 더 안정적으로 학습할 수 있도록 하기 위함이다. 네트워크는 일반적으로 입력 데이터의 값이 작을수록 더 잘 작동하기 때문이다.
# size 매개변수는 네트워크에 입력되는 이미지의 크기를 지정한다. YOLO는 (320, 320), (416, 416), (609, 609)의 세 가지 크기만을 지원하며 크기가 커질수록 
# 정확도가 더 높아지지만 동시에 속도는 더 느려진다.
blob: MatLike = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True, crop=False)

# blob을 네트워크의 Input 데이터로 설정한다.
net.setInput(blob)

# 입력된 blob을 네트워크에서 순방향으로 전파시킨다.
outputList: Sequence[MatLike] = net.forward(outputLayerList)


classIdList: list[intp] = []
confidenceList: list[float] = []
boxList: list[list[int, int, int, int]] = []

# 정보를 화면에 표시하는 작업
for output in outputList:
    for detection in output:
        # YOLO의 출력 레이어 배열(ndarray)은 다음과 같은 구조를 가진다.
        # 1. 인덱스 0, 1, 2, 3은 bounding box의 중심 좌표 (center_x, center_y, width, height)를 나타낸다.
        # 2. 인덱스 4 이후는 각 클래스에 대한 신뢰도(confidence)를 나타낸다.
        positions: ndarray[float32] = detection[:4]
        scores: ndarray[float32] = detection[5:]

        # 각각의 클래스에 대한 신뢰도 중 가장 신뢰도가 높은 값의 인덱스를 가져온다.
        # 여기서 argmax 메소드는 f(x)의 최대값을 만들어 주는 입력 x를 반환시킨다.
        # 즉, list.index(max(list)) 와 같다.
        class_id: intp = np.argmax(scores)

        # 해당 객체와 가장 신뢰도가 높은 클래스의 신뢰도를 가져온다.
        confidence: float32 = scores[class_id]

        # 만약 신뢰도가 임계값보다 크다면 객체가 감지된 것 (조절 가능)
        # 임계값은 0에서 1 사이의 값을 가지며, 1에 가까울수록 정확도가 높아지고
        # 0에 가까울수록 정확도는 낮아지지만 탐지되는 Object의 수는 많아진다.
        if confidence > CONFIDENCE_CRITICAL_NUMBER:

            # 이미지 크기에 비례하여 이미지상의 해당 오브젝트의 중심 x, y 좌표값을 저장한다.
            center_x = int(positions[0] * width)
            center_y = int(positions[1] * height)

            # 해당 오브젝트의 가로 길이, 세로 길이도 이미지의 크기에 비례하여 저장한다.
            w = int(positions[2] * width)
            h = int(positions[3] * height)

            # 박스를 생성할 때 기준삼을 x, y 좌표를 저장한다.
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # 이미지에 출력하기 위해 해당 정보들을 리스트에 저장해준다.
            classIdList.append(class_id)
            confidenceList.append(float(confidence))
            boxList.append([x, y, w, h])

# Non-Maximum Suppression (NMS)을 사용하여 중복된 경계 상자를 제거하고 최종적으로 선택된 경계 상자의 인덱스를 반환한다.
result: Sequence[int] = cv2.dnn.NMSBoxes(boxList, confidenceList, 0.5, 0.4)

# Object Class Name을 표시할 때 사용할 글자의 Font 정보
font: int = cv2.FONT_HERSHEY_PLAIN

for i in range(len(boxList)):
    if i in result:
        # box의 위치, 크기정보를 가져온다.
        x, y, w, h = boxList[i]

        # 해당 Object의 Class Name을 불러온다.
        label = str(classList[classIdList[i]])
        
        # Box, Font에 사용할 랜덤한 색상
        color = colorList[i]

         # Box를 그려넣고 Class Name을 적어넣는다.
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

# 이미지 출력
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
