FROM python:3.9
WORKDIR /app
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN mkdir video
RUN mkdir models
RUN mkdir utils
COPY video/DogDog8.mp4 video/DogDog8.mp4
COPY models/emotion_classifier_resnet.pkl models/emotion_classifier_resnet.pkl
COPY models/face_detector_yolov8n.pt models/face_detector_yolov8n.pt
COPY utils/model_class.py utils/model_class.py
COPY mvp.py mvp.py
EXPOSE 8502
HEALTHCHECK CMD curl --fail http://localhost:8502/_stcore/health
ENTRYPOINT ["streamlit", "run", "mvp.py", "--server.port=8502", "--server.address=0.0.0.0"]
