# Инструкция по запуску сервиса
## Шаг 1
Склонируйте репозиторий командой:
```
git clone https://github.com/kazars24/mfdp-emotion-detection.git
```
И перейдите в папку mfdp-emotion-detection:
```
cd mfdp-emotion-detection
```
## Шаг 2
Для успешного использования сервиса вам необходимо установить себе на компьютер [Docker](https://www.docker.com/).
После чего выполнить следующую команду:
```
docker build -t mfdp .
```
## Шаг 3
И наконец, для запуска приложения нужно выполнить команду:
```
docker run -p 8502:8502 mfdp
```
После чего к сервису можно перейти по одному их этих хостов:
* URL: http://0.0.0.0:8502
* URL: http://localhost:8502/

