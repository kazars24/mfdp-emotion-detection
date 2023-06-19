FROM python:3.9
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install zlib numpy
RUN pip install opencv-python==4.5.5.64
RUN pip install -r requirements.txt
COPY DogDog8.mp4 DogDog8.mp4
COPY mvp.py mvp.py
EXPOSE 8502
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "mvp.py", "--server.port=8502", "--server.address=0.0.0.0"]
