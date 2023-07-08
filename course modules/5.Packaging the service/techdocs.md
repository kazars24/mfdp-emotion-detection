# Техническая документация
## Описание
Данная техническая документация описывает модули приложения для распознавания эмоций в маркетинговых исследованиях. Приложение использует различные библиотеки для обработки видео, распознавания эмоций на лицах и визуализации результатов.
## Зависимости
Для успешной работы приложения необходимо установить следующие зависимости:
* Python 3.7 или выше
* av==10.0.0
* facenet-pytorch==2.5.3
* ffmpeg==1.4
* matplotlib==3.7.1
* mediapipe==0.8.9.1
* numpy==1.24.3
* opencv_contrib_python==4.7.0.72
* opencv_python==4.7.0.72
* Pillow==9.5.0
* plotly==5.14.1
* seaborn==0.12.2
* streamlit==1.22.0
* streamlit_webrtc==0.45.1
* torch==2.0.1
* torchvision==0.15.2
* ultralytics==8.0.123
* watchdog==3.0.0

Установка зависимостей может быть выполнена с использованием менеджера пакетов pip следующим образом:
```
pip install -r requirements.txt
```
## Модули
### Модуль 1: Основное приложение (файл mvp.py)
Основной модуль приложения, который содержит код для запуска веб-интерфейса и обработки видео с использованием веб-камеры. Включает следующие функции:

* __callback(frame)__: Функция, которая выполняет распознавание эмоций на кадре видео. Используется как параметр video_frame_callback в webrtc_streamer.
* __make_barchart(placeholder)__: Функция для создания столбчатой диаграммы, визуализирующей количество обнаруженных эмоций. Записывает значения EMOTION_COUNT в session_state.
* __make_linegraph(placeholder)__: Функция для создания линейного графика, отображающего эмоциональную динамику. Записывает значения LINE_DATA в session_state.
```
import time

import av
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from utils.model_class import EmoRecNet

st.title("My First Data Project 2: Распознавание эмоций в маркетинговых исследованиях")

EMOTION_COUNT = {'angry': 0,
                 'disgust': 0,
                 'fear': 0,
                 'happy': 0,
                 'sad': 0,
                 'surprise': 0,
                 'neutral': 0}
LINE_DATA = []
if 'emotion_count_dict' not in st.session_state:
    st.session_state['emotion_count_dict'] = {'angry': 0,
                                              'disgust': 0,
                                              'fear': 0,
                                              'happy': 0,
                                              'sad': 0,
                                              'surprise': 0,
                                              'neutral': 0}
if 'emotion_dinamics' not in st.session_state:
    st.session_state['emotion_dinamics'] = []
if 'analyze' not in st.session_state:
    st.session_state['analyze'] = False

model = EmoRecNet()

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = Image.fromarray(img)

    captured_emotions = model.predict(img)


    for face in captured_emotions:
        box = face['bbox']
        label = face['emotion']
        EMOTION_COUNT[label] += 1
        if label in ['happy', 'surprise']:
            LINE_DATA.append(1)
        elif label in ['angry', 'disgust', 'fear', 'sad']:
            LINE_DATA.append(-1)
        elif label == 'neutral':
            LINE_DATA.append(0)

        xmin, ymin, xmax, ymax = box
        nimg = np.array(img)
        cv2_img = nimg
        cv2.rectangle(cv2_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(
            cv2_img,
            label,
            (xmin, ymin - 15 if ymin - 15 > 15 else ymin + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )
    np_img = np.asarray(cv2_img)

    return av.VideoFrame.from_ndarray(np_img, format="bgr24")


def make_barchart(placeholder):
    if st.session_state['analyze']:
        st.session_state['emotion_count_dict'] = EMOTION_COUNT
    
    emotions = list(EMOTION_COUNT.keys())
    counts = list(EMOTION_COUNT.values())
    data = {'Emotion': emotions, 'Count': counts}
    df = pd.DataFrame(data)
    fig = px.bar(df, x='Emotion', y='Count', color='Emotion')
    fig.update_layout(title='Emotion count')
    placeholder.write(fig)


def make_linegraph(placeholder):
    if st.session_state['analyze']:
        st.session_state['emotion_dinamics'] = LINE_DATA

    fig = px.line(LINE_DATA)
    fig.update_layout(xaxis_title="Frame", yaxis_title="Emotion", title='Emotion dinamics')
    placeholder.write(fig)


st.write("Шаг 1. Включите камеру, нажав на красную кнопку START")
webrtc_streamer(key="example",
                mode=WebRtcMode.SENDRECV,
                video_frame_callback=callback,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True)

st.write("Шаг 2. Запустите видео")
video_file = open('video/DogDog8.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)

st.write("Шаг 3. Запустите анализ эмоций, нажав кнопку ANALYZE EMOTION")
EMOTION_COUNT = {'angry': 0,
                 'disgust': 0,
                 'fear': 0,
                 'happy': 0,
                 'sad': 0,
                 'surprise': 0,
                 'neutral': 0}
LINE_DATA = []
placeholder_1 = st.empty()
placeholder_2 = st.empty()
start_button = st.empty()
stop_flag = True

if start_button.button('ANALYZE EMOTION', key='start'):
    start_button.empty()
    st.session_state['analyze'] = True
    if not st.button('Stop', key='stop'):
        while True:
            make_barchart(placeholder_1)
            make_linegraph(placeholder_2)
            time.sleep(0.1)
    else:
        emotion_count_data = st.session_state['emotion_count_dict']
        emotions = list(emotion_count_data.keys())
        counts = list(emotion_count_data.values())
        data_dict = {'Emotion': emotions, 'Count': counts}
        df = pd.DataFrame(data_dict)
        fig_bar = px.bar(df, x='Emotion', y='Count', color='Emotion')
        fig_bar.update_layout(title='Emotion count')
        st.plotly_chart(fig_bar, use_container_width=True)

        emotion_dinamics_data = st.session_state['emotion_dinamics']
        fig_line = px.line(emotion_dinamics_data)
        fig_line.update_layout(xaxis_title="Frame", yaxis_title="Emotion", title='Emotion dinamics')
        st.plotly_chart(fig_line, use_container_width=True)


```
### Модуль 2: Класс используемого ML-модуля (файл utils/model_class.py)
Класс __EmoRecNet__ реализует распознавание эмоций на лицах с использованием модели YOLOv8n для обнаружения лиц и модели ResNet для классификации эмоций. Класс обеспечивает обработку изображений, обнаружение лиц, классификацию эмоций и возвращает результаты в виде словарей с информацией о координатах баундин бокса лица в фомате [xmin, ymin, xmax, ymax] и классе эмоции.
Данный класс содержит следующие методы:
* **__init__(self)**: инициализирует объект класса EmoRecNet, загружает модели YOLO и ResNet, переносит модели на устройство (GPU или CPU) в зависимости от доступности.
* **models_to_device(self)**: переносит модель ResNet на устройство.
* **detect_face(self, frame)**: выполняет обнаружение лиц на кадре с помощью модели YOLO, возвращает список границ лиц (bboxes), где каждая граница представлена в формате [xmin, ymin, xmax, ymax].
* **classify_emotions(self, frame, bboxes)**: выполняет классификацию эмоций на лицах, ограниченных границами (bboxes), с помощью модели ResNet; возвращает класс эмоций (emotions), где каждый представлен целочисленным значением.
* **predict(self, frame)**: выполняет обнаружение лиц и классификацию эмоций на кадре видео или изображении; возвращает результаты распознавания эмоций в виде списка словарей, где каждый словарь содержит информацию о границах лица (bbox) и классе эмоции (emotion).
* **get_labels(self)**: возвращает словарь с соответствием между классами эмоций и их текстовыми наименованиями.
```
from ultralytics import YOLO
import pickle
import torch
from torchvision import transforms


class EmoRecNet(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = YOLO('models/face_detector_yolov8n.pt')
        with open('models/emotion_classifier_resnet.pkl', 'rb') as file:
            self.classifier = pickle.load(file)
        self.models_to_device()

    def models_to_device(self):
        self.classifier = self.classifier.to(self.device).eval()

    def detect_face(self, frame):
        width, height = frame.size
        bboxes = []
        faces = self.detector.predict(source=frame, device=self.device, max_det=1)
        for face in faces:
            bbox_pred = face.boxes.xyxy
            if bbox_pred.nelement() != 0:
                xmin, ymin, xmax, ymax = bbox_pred[0].cpu().tolist()
                xmin = int(max(0, xmin - 5))
                ymin = int(max(0, ymin - 5))
                xmax = int(min(width, xmax + 5))
                ymax = int(min(height, ymax + 5))
                # bboxes.append(bbox_pred[0].cpu().tolist())
                bboxes.append([xmin, ymin, xmax, ymax])
        return bboxes

    def classify_emotions(self, frame, bboxes):
        face_transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        emotions = []
        for box in bboxes:
            croped_face = frame.crop(box)
            croped_face = face_transform(croped_face)
            croped_face = croped_face.to(self.device)
            croped_face = torch.unsqueeze(croped_face, 0)
            pred_class = int(torch.argmax(self.classifier(croped_face), dim=1))
            emotions.append(pred_class)
        return emotions

    def predict(self, frame):
        labels = self.get_labels()
        bboxes = self.detect_face(frame)
        emotions = self.classify_emotions(frame, bboxes)
        result = []
        for i in range(len(emotions)):
            res_dict = {'bbox': bboxes[i], 'emotion': labels[emotions[i]]}
            result.append(res_dict)
        return result

    def get_labels(self):
        return {
            0: 'angry',
            1: 'disgust',
            2: 'fear',
            3: 'happy',
            4: 'neutral',
            5: 'sad',
            6: 'surprise'
        }

```

## Использование
1. Установите необходимые зависимости, указанные в разделе "Зависимости".
2. Подключите модули и функции к вашему проекту или скрипту.
3. Запустите основное приложение, которое будет отображать видеопоток и результаты распознавания эмоций.

## Заключение
Техническая документация предоставляет описание модулей и функций приложения для распознавания эмоций в маркетинговых исследованиях. Пользуйтесь документацией для успешной интеграции и использования приложения.
