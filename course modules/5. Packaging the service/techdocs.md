# Техническая документация
## Описание
Данная техническая документация описывает модули приложения для распознавания эмоций в маркетинговых исследованиях. Приложение использует библиотеки Streamlit, streamlit-webrtc, av, cv2, os, time, fer, matplotlib, seaborn и plotly для обработки видео, распознавания эмоций на лицах и визуализации результатов.
## Зависимости
Для успешной работы приложения необходимо установить следующие зависимости:
* Python 3.7 или выше
* streamlit
* streamlit-webrtc
* av
* opencv-python (cv2)
* fer
* matplotlib
* seaborn
* plotly
* ffmpeg

Установка зависимостей может быть выполнена с использованием менеджера пакетов pip следующим образом:
```
pip install streamlit streamlit-webrtc av opencv-python fer matplotlib seaborn plotly
```
## Модули
### Модуль 1: Основное приложение
```
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import os
import time
from fer import FER
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Установка пути к исполняемому файлу FFmpeg
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

st.title("My First Data Project 2: Распознавание эмоций в маркетинговых исследованиях")

EMOTION_COUNT = {'angry': 0,
                 'disgust': 0,
                 'fear': 0,
                 'happy': 0,
                 'sad': 0,
                 'surprise': 0,
                 'neutral': 0}
LINE_DATA = []

def callback(frame):
    # Создание экземпляра детектора эмоций
    detector = FER(mtcnn=True)
    img = frame.to_ndarray(format="bgr24")

    # Распознавание эмоций на изображении
    captured_emotions = detector.detect_emotions(img)

    if captured_emotions:
        label = max(captured_emotions[0]['emotions'], key=captured_emotions[0]['emotions'].get)
        EMOTION_COUNT[label] += 1

        # Обработка эмоциональной оценки для графика
        if label in ['happy', 'surprise']:
            LINE_DATA.append(1)
        elif label in ['angry', 'disgust', 'fear', 'sad']:
            LINE_DATA.append(-1)
        elif label == 'neutral':
            LINE_DATA.append(0)

        # Извлечение координат лица и оценки вероятности эмоции
        xmin, ymin, w, h = captured_emotions[0]['box']
        xmax = w + xmin
        ymax = h + ymin
        score = captured_emotions[0]['emotions'][label]

        caption = f"{label}: {round(score * 100, 2)}%"

        # Отрисовка прямоугольника и подписи на изображении
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(
            img,
            caption,
            (xmin, ymin - 15 if ymin - 15 > 15 else ymin + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )
    return av.VideoFrame.from_ndarray(img, format="bgr24")

def make_barchart():
    # Создание столбчатой диаграммы для визуализации количества эмоций
    fig = plt.figure(figsize=(10, 5))
    sns.barplot(x=list(EMOTION_COUNT.keys()), y=list(EMOTION_COUNT.values()))
    placeholder_1.write(fig)

def make_linegraph():
    # Создание линейного графика для визуализации эмоциональной динамики
    fig = px.line(LINE_DATA)
    fig.update_layout(xaxis_title="frame", yaxis_title="emotion", title='Emotion graph')
    placeholder_2.write(fig)

st.write("Шаг 1. Включите камеру, нажав на красную кнопку START")
webrtc_streamer(key="example",
                video_frame_callback=callback,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True)

st.write("Шаг 2. Запустите видео")
video_file = open('DogDog8.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)

placeholder_1 = st.empty()
placeholder_2 = st.empty()
start_button = st.empty()

EMOTION_COUNT = {'angry': 0,
                 'disgust': 0,
                 'fear': 0,
                 'happy': 0,
                 'sad': 0,
                 'surprise': 0,
                 'neutral': 0}
LINE_DATA = []

if start_button.button('Enable emotion recognition', key='start'):
    start_button.empty()

    if st.button('Stop', key='stop'):
        st.stop()

    while True:
        make_barchart()
        make_linegraph()
        time.sleep(0.1)
```
### Модуль 2: Функция обратного вызова (callback)
```
def callback(frame):
    # Создание экземпляра детектора эмоций
    detector = FER(mtcnn=True)
    img = frame.to_ndarray(format="bgr24")

    # Распознавание эмоций на изображении
    captured_emotions = detector.detect_emotions(img)

    if captured_emotions:
        label = max(captured_emotions[0]['emotions'], key=captured_emotions[0]['emotions'].get)
        EMOTION_COUNT[label] += 1

        # Обработка эмоциональной оценки для графика
        if label in ['happy', 'surprise']:
            LINE_DATA.append(1)
        elif label in ['angry', 'disgust', 'fear', 'sad']:
            LINE_DATA.append(-1)
        elif label == 'neutral':
            LINE_DATA.append(0)

        # Извлечение координат лица и оценки вероятности эмоции
        xmin, ymin, w, h = captured_emotions[0]['box']
        xmax = w + xmin
        ymax = h + ymin
        score = captured_emotions[0]['emotions'][label]

        caption = f"{label}: {round(score * 100, 2)}%"

        # Отрисовка прямоугольника и подписи на изображении
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(
            img,
            caption,
            (xmin, ymin - 15 if ymin - 15 > 15 else ymin + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )
    return av.VideoFrame.from_ndarray(img, format="bgr24")
```
### Модуль 3: Функция создания столбчатой диаграммы (make_barchart)
```
def make_barchart():
    # Создание столбчатой диаграммы для визуализации количества эмоций
    fig = plt.figure(figsize=(10, 5))
    sns.barplot(x=list(EMOTION_COUNT.keys()), y=list(EMOTION_COUNT.values()))
    placeholder_1.write(fig)
```
### Модуль 4: Функция создания линейного графика (make_linegraph)
```
def make_linegraph():
    # Создание линейного графика для визуализации эмоциональной динамики
    fig = px.line(LINE_DATA)
    fig.update_layout(xaxis_title="frame", yaxis_title="emotion", title='Emotion graph')
    placeholder_2.write(fig)
```
## Использование
1. Установите необходимые зависимости, указанные в разделе "Зависимости".
2. Подключите модули и функции к вашему проекту или скрипту.
3. Запустите основное приложение, которое будет отображать видеопоток и результаты распознавания эмоций.

## Примечания
* Возможно потребуется указать правильный путь к исполняемому файлу FFmpeg (ffmpeg) в переменной окружения IMAGEIO_FFMPEG_EXE для корректной работы с видео.
* Приложение поддерживает как использование камеры для обработки видео в режиме реального времени, так и загрузку видеофайла для обработки.
* Для остановки приложения можно нажать кнопку "Stop".
* График столбчатой диаграммы будет отображаться в placeholder_1, а линейного графика - в placeholder_2.

## Заключение
Техническая документация предоставляет описание модулей и функций приложения для распознавания эмоций в маркетинговых исследованиях. Пользуйтесь документацией для успешной интеграции и использования приложения.
