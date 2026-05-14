from ultralytics import YOLO
import cv2
import easyocr

# YOLO модель
model = YOLO('runs/detect/train/weights/best.pt')

# OCR
reader = easyocr.Reader(['ru'], gpu=True)

# Видео
cap = cv2.VideoCapture('')

DISPLAY_SCALE = 0.5

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Поворот
    rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Детекция
    results = model(rotated)

    annotated = rotated.copy()

    for result in results:

        boxes = result.boxes

        for box in boxes:

            # Координаты
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Вырезаем ценник
            crop = rotated[y1:y2, x1:x2]

            # OCR
            text_results = reader.readtext(crop)

            detected_text = ""

            for t in text_results:
                detected_text += t[1] + " "

            detected_text = detected_text.strip()

            # Рисуем прямоугольник
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Вывод текста
            cv2.putText(
                annotated,
                detected_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

    # Масштабирование окна
    h, w = annotated.shape[:2]

    new_w = int(w * DISPLAY_SCALE)
    new_h = int(h * DISPLAY_SCALE)

    display_frame = cv2.resize(
        annotated,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    cv2.imshow('YOLO + OCR', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()