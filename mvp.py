import time

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer

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
    fig = plt.figure(figsize=(10, 5))
    sns.barplot(x=list(EMOTION_COUNT.keys()), y=list(EMOTION_COUNT.values()))
    placeholder.write(fig)


def make_linegraph(placeholder):
    fig = px.line(LINE_DATA)
    fig.update_layout(xaxis_title="frame", yaxis_title="emotion", title='Emotion graph')
    placeholder.write(fig)


st.write("Шаг 1. Включите камеру, нажав на красную кнопку START")
webrtc_streamer(key="example",
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
    if not st.button('Stop', key='stop'):
        while True:
            make_barchart(placeholder_1)
            make_linegraph(placeholder_2)
            time.sleep(0.1)
