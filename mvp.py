import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import os
import time

os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
from fer import FER
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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
    detector = FER(mtcnn=True)
    img = frame.to_ndarray(format="bgr24")

    captured_emotions = detector.detect_emotions(img)

    if captured_emotions:
        label = max(captured_emotions[0]['emotions'], key=captured_emotions[0]['emotions'].get)
        EMOTION_COUNT[label] += 1
        if label in ['happy', 'surprise']:
            LINE_DATA.append(1)
        elif label in ['angry', 'disgust', 'fear', 'sad']:
            LINE_DATA.append(-1)
        elif label == 'neutral':
            LINE_DATA.append(0)
        xmin, ymin, w, h = captured_emotions[0]['box']
        xmax = w + xmin
        ymax = h + ymin
        score = captured_emotions[0]['emotions'][label]

        caption = f"{label}: {round(score * 100, 2)}%"

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
    fig = plt.figure(figsize=(10, 5))
    sns.barplot(x=list(EMOTION_COUNT.keys()), y=list(EMOTION_COUNT.values()))
    placeholder_1.write(fig)


def make_linegraph():
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

# TODO: сделать в конце показ конечных графиков