# Price Tag Detection System

Система для детекции и распознавания магазинных ценников с использованием YOLO, ByteTrack и Ollama/Qwen2.5-VL.

---

# Возможности

* Детекция ценников через YOLO
* Трекинг объектов через ByteTrack
* OCR и извлечение данных через Qwen2.5-VL
* Поддержка видео и изображений
* Автоматическая классификация типов ценников
* Экспорт результата в CSV
* Поддержка GPU (CUDA)

---

# Установка

## 1. Установка Python

⚠️ Обязательно используйте Python 3.11

Проверка версии:

```bash
python --version
```

Должно быть:

```bash
Python 3.11.x
```

---

## 2. Создание виртуального окружения

```bash
python -m venv venv
```

Активация:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## 3. Установка CUDA PyTorch

Для CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 4. Установка YOLO

```bash
pip install ultralytics
```

---

## 5. Установка остальных зависимостей

```bash
pip install opencv-python pillow pyzbar numpy requests
```

Для работы preview-окна рекомендуется:

```bash
pip install opencv-python
```

Если используется серверная среда без GUI:

```bash
pip install opencv-python-headless
```

---

# Проверка GPU

Создайте файл `test_cuda.py`:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Запуск:

```bash
python test_cuda.py
```

Если всё работает корректно — увидите название вашей видеокарты.

---

# Установка Ollama

## 1. Скачать Ollama

Скачать релизы:

https://github.com/ollama/ollama/releases

Для Windows:

https://github.com/ollama/ollama/releases/download/v0.13.3/OllamaSetup.exe

---

## 2. Запуск Ollama

⚠️ Перед запуском убедитесь что десктопная версия Ollama полностью закрыта.

В терминале:

```bash
ollama serve
```

---

## 3. Загрузка модели

⚠️ Команду нужно выполнять В ДРУГОМ окне терминала.

```bash
ollama pull qwen2.5vl:7b
```

---

## 4. Проверка работы модели

```bash
ollama run qwen2.5vl:7b "Привет! ты работаешь?"
```

Если всё работает — модель ответит текстом.

---

# Структура проекта

```text
project/
│
├── detect.py
├── video_detect.py
├── runs/
│   └── detect/
│       └── train-2/
│           └── weights/
│               └── best.pt
│
├── examples/
│   ├── standard.jpg
│   ├── weight.jpg
│   ├── wholesale.jpg
│   ├── wine.jpg
│   ├── shelftaker.jpg
│   └── simple.jpg
│
└── result.csv
```

---

# Запуск распознавания изображений

```bash
python detect.py ./images -o result.csv
```

Или:

```bash
python detect.py image1.jpg image2.png -o result.csv
```

---

# Запуск распознавания видео

```bash
python video_detect.py input.mp4
```

С указанием CSV:

```bash
python video_detect.py input.mp4 -o result.csv
```

Без preview-окна:

```bash
python video_detect.py input.mp4 --no-preview
```

Без коррекции дисторсии:

```bash
python video_detect.py input.mp4 --no-undistort
```

---

# Параметры запуска

## video_detect.py

| Параметр            | Описание                     |
| ------------------- | ---------------------------- |
| `-o`                | Выходной CSV                 |
| `--model`           | Путь к YOLO модели           |
| `--ollama-model`    | Название Ollama модели       |
| `--ollama-host`     | Адрес Ollama                 |
| `--reprocess-after` | Повторная обработка track    |
| `--no-preview`      | Отключить preview            |
| `--no-undistort`    | Отключить fisheye correction |

---

# Формат результата CSV

Система сохраняет:

* название товара
* цены
* скидки
* barcode
* QR данные
* SKU
* координаты bbox
* цвет рамки
* timestamp кадра

---

# Поддерживаемые типы ценников

* standard
* weight
* wholesale
* wine
* shelftaker
* simple

---

# Few-shot примеры

Система поддерживает эталонные изображения в папке `examples/`.

Если папка отсутствует — система продолжит работать без few-shot.

---

# Частые ошибки

## Ошибка CUDA / Torch

```text
RuntimeError: module compiled against ABI version
```

Решение:

```bash
pip uninstall numpy
pip install numpy==1.26.4
```

---

## OpenCV preview не работает

```text
cvShowImage function is not implemented
```

Решение:

```bash
pip uninstall opencv-python-headless
pip install opencv-python
```

---

## Ollama не отвечает

Проверьте что запущено:

```bash
ollama serve
```

И что модель скачана:

```bash
ollama list
```

---

# Производительность

## CPU

* ~1–3 минуты на ценник
* высокая нагрузка на процессор

## GPU

Рекомендуется использовать NVIDIA GPU с CUDA.

---

# Рекомендуемые характеристики

| Компонент    | Рекомендация |
| ------------ | ------------ |
| Python       | 3.11         |
| RAM          | 16 GB+       |
| GPU          | NVIDIA CUDA  |
| VRAM         | 8 GB+        |
| Ollama Model | qwen2.5vl:7b |

---

# Основной запуск

После установки всего окружения:

```bash
python detect.py . -o result.csv
```

или

```bash
python video_detect.py input.mp4 -o result.csv
```

---

# Важно

* Ollama должна быть запущена ДО старта Python-скриптов
* Используйте отдельный терминал для `ollama serve`
* Для стабильной работы рекомендуется GPU
* Не используйте Python 3.12+ — возможны проблемы с зависимостями
