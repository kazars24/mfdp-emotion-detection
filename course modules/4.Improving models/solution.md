# Отчет о проведенных экспериментах
Весь процесс экспериментов и его описание доступны в файле [experiments_v2.ipynb](https://github.com/kazars24/mfdp-emotion-detection/blob/main/course%20modules/4.Improving%20models/experiments_v2.ipynb), а тут можно ознакомиться с полученными результатами.
## Модели и данные
**Детекция лица на изображении.** Для этой задачи были использованы:
1. MTCNN из библиотеки facenet_pytorch;
2. MTCNN из библиотеки fer;
3. каскад Хаара из библиотеки fer;
4. YOLOv8.

**Распознавание эмоций.** Были рассмотрены следующие модели:
1. InceptionResnetV1(pretrained='vggface2') с файнтюнингом на RAF Face Database;
2. InceptionResnetV1(pretrained='casia-webface') с файнтюнингом на RAF Face Database;
3. InceptionResnetV1 с обучением на RAF Face Database.

Проверка моделей осуществлялась на тестовой части датасета [RAF Face Database](http://www.whdeng.cn/raf/model1.html).

Используемый инструменты версионирования экспериментов - clearML.

## Результаты
### Детекция лиц
| Алгоритм | Ср. IoU | Ср. время | Не найдено лиц |
|----------|----------|----------|----------------|
| MTCNN (facenet_pytorch) | 0.82 | 0.18 | 2.5% |
| MTCNN (fer) | 0.72 | 0.23 | 13.92% |
| Каскад Хаара (fer) | 0.51 | 0.1 | 20.96% |
| YOLOv8n | 0.9 | 0.03 | 0.03% |

YOLOv8n показала себя лучше других по всем показателям, поэтому этот вариант и возьмем.

### Классификация эмоций
Результаты на валидационной выборке
| Модель | Loss | FPS | Accuracy|
|----------|----------|----------|----------------|
| InceptionResnetV1(pretrained='vggface2')      | 0.6863 | 796.3889 | 0.8472 |
| InceptionResnetV1(pretrained='casia-webface') | 0.5098 | 1609.1836 | 0.8576 |
| InceptionResnetV1(pretrained=None)            | 0.5689 | 848.1443 | 0.8048 |

Тут стоить выбрать InceptionResnetV1(pretrained='casia-webface') с файнтюнингом на RAF Face Database в сравнении со всеми показателями. Особенно эта модель выигрывает в fps.

## Вывод
Таким образом, наш пайплайн по распознаванию эмоций будет состоять из YOLOv8n и InceptionResnetV1(pretrained='casia-webface').
