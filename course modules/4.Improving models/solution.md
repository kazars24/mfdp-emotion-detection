# Отчет о проведенных экспериментах
Весь процесс экспериментов и его описание доступны в файле experiments.ipynb, а тут можно ознакомиться с полученными результатами.
## Модели и данные
**Детекция лица на изображении.** Для этой задачи были использованы MTCNN из библиотеки facenet_pytorch и MTCNN из библиотеки fer.
**Распознавание эмоций.**Были рассмотрены следующие модели:
1.   Алгоритм из fer;
2.   InceptionResnetV1(pretrained='vggface2') с файнтюнингом на Facial Affect Dataset(short);
3. InceptionResnetV1(pretrained='casia-webface') с файнтюнингом на Facial Affect Dataset(short);
4. InceptionResnetV1 с обучением на Facial Affect Dataset(short).

Проверка моделей осуществлялась на валидационной части датасета Facial Affect Dataset(short) и на тестовой части датасета RAF Face Database (к нему мне, к сожалению, предоставили доступ накануне дедлайна).

Используемый инструменты версионирования экспериментов - clearML.

## Результаты
### Facial Affect Dataset(short)
| Алгоритм | Сек/кадр | Accuracy | Не найдено лиц |
|----------|----------|----------|----------------|
| MTCNN + InceptionResnetV1 | 0.04 | 0.32 | 0.19% |
| Row 2    | Cell 5   | Cell 6   |
| Row 3    | Cell 8   | Cell 9   |
