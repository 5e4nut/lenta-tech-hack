import cv2
import os
from ultralytics import YOLO
import easyocr
import numpy as np

# ---------------- НАСТРОЙКИ ----------------
VIDEO_PATH = 'test.mp4'
MODEL_PATH = 'runs/detect/train-2/weights/best.pt'
OUTPUT_VIDEO = 'result.mp4'
MIN_CONF = 0.5
PAD = 10

# ---------------- ПРОВЕРКА ФАЙЛОВ ----------------
if not os.path.exists(VIDEO_PATH):
    print(f"❌ Видео не найдено: {VIDEO_PATH}")
    exit()
if not os.path.exists(MODEL_PATH):
    print(f"❌ Модель не найдена: {MODEL_PATH}")
    exit()

# ---------------- ЗАГРУЗКА ВИДЕО ----------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Не удалось открыть видео")
    exit()

# Получаем параметры ИСХОДНОГО видео
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
print(f" Исходное видео: {orig_w}x{orig_h}, {fps} FPS")

# Читаем первый кадр для поворота и получения реальных размеров после трансформации
ret, first_frame = cap.read()
if not ret:
    print("❌ Видео пустое или битое")
    exit()

#  ПОВОРОТ НА 270° ПО ЧАСОВОЙ (в OpenCV это 90° ПРОТИВ часовой)
frame = cv2.rotate(first_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
height, width = frame.shape[:2]
print(f"🔄 Видео повернуто. Рабочие размеры: {width}x{height}")

# Сбрасываем указатель на начало видео
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ---------------- МОДЕЛИ ----------------
print("⏳ Загрузка YOLO...")
model = YOLO(MODEL_PATH)
print("⏳ Загрузка EasyOCR...")
reader = easyocr.Reader(['ru'], gpu=True, verbose=False)
print("✅ Модели загружены. Начинаю обработку...\n")

# ---------------- VIDEO WRITER (с новыми размерами) ----------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))


# ---------------- ПРЕПРОЦЕССИНГ ДЛЯ OCR ----------------
def preprocess_crop(crop):
    if crop.size == 0: return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    if gray.shape[1] < 300:
        scale = 300 / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(thresh) < 100:
        thresh = cv2.bitwise_not(thresh)
    return thresh


# ---------------- ОСНОВНОЙ ЦИКЛ ----------------
frame_count = 0
print("=" * 60)
print("Нажмите 'q' для выхода, 's' для сохранения кадра")
print("=" * 60 + "\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 🔄 Поворачиваем каждый кадр сразу после чтения
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    print(f"\n{'=' * 60}")
    print(f"КАДР {frame_count}")
    print(f"{'=' * 60}")

    # Детекция YOLO (теперь работает на правильно ориентированном кадре)
    results = model(frame, conf=MIN_CONF)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        for i, (box, conf) in enumerate(zip(boxes, confs)):
            x1, y1, x2, y2 = map(int, box)

            # Padding с учётом границ повёрнутого кадра
            x1 = max(0, x1 - PAD)
            y1 = max(0, y1 - PAD)
            x2 = min(width, x2 + PAD)
            y2 = min(height, y2 + PAD)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue

            processed = preprocess_crop(crop)

            print(f"\n ЦЕННИК #{i + 1} (conf: {conf:.2f})")
            print(f"Координаты: [{x1}, {y1}, {x2}, {y2}]")
            print("-" * 40)

            if processed is not None:
                ocr_results = reader.readtext(processed, paragraph=False, detail=1)

                if ocr_results:
                    print("РАСПОЗНАННЫЙ ТЕКСТ:")
                    full_text = []
                    for item in ocr_results:
                        text = item[1]
                        ocr_conf = item[2]
                        if ocr_conf > 0.3:
                            print(f"  ✓ {text} (conf: {ocr_conf:.2f})")
                            full_text.append(text)
                    print(f"\nПОЛНЫЙ ТЕКСТ: {' | '.join(full_text)}")
                else:
                    print("⚠ Текст не распознан")
            else:
                print("⚠ Ошибка обработки кропа")

            # Отрисовка
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Tag #{i + 1}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Price Tag Detection (Rotated 270°)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\n⏹ Остановка пользователем")
        break
    elif key == ord('s'):
        cv2.imwrite(f"frame_{frame_count}.png", frame)
        print(f"💾 Кадр {frame_count} сохранён")

# ---------------- ЗАВЕРШЕНИЕ ----------------
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\n{'=' * 60}")
print(f"✅ ОБРАБОТКА ЗАВЕРШЕНА")
print(f"Кадров обработано: {frame_count}")
print(f"Результат: {OUTPUT_VIDEO}")