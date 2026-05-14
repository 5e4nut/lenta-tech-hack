from ultralytics import YOLO
import cv2
import easyocr

model = YOLO('runs/detect/train-2/weights/best.pt')
reader = easyocr.Reader(['ru'], gpu=True)

cap = cv2.VideoCapture('test.mp4')
DISPLAY_SCALE = 0.5
MIN_BOX_AREA = 50

frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    results = model(rotated)
    annotated = rotated.copy()

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area < MIN_BOX_AREA:
                continue

            crop = rotated[y1:y2, x1:x2]

            # --- УЛУЧШЕНИЕ РАСПОЗНАВАНИЯ ---
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)          # в серый
            gray = cv2.equalizeHist(gray)                          # выравнивание контраста
            # Дополнительно: бинаризация (раскомментировать если нужно)
            # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            #                              cv2.THRESH_BINARY, 11, 2)

            # OCR уже на сером изображении
            text_results = reader.readtext(gray)
            detected_text = " ".join([t[1] for t in text_results]).strip()

            if detected_text:
                print(f"[Кадр {frame_counter}] Ценник ({x1},{y1},{x2},{y2}): {detected_text}")
            else:
                print(f"[Кадр {frame_counter}] Ценник ({x1},{y1},{x2},{y2}): текст не распознан")

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Отображение
    h, w = annotated.shape[:2]
    new_w, new_h = int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)
    display_frame = cv2.resize(annotated, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.imshow('YOLO + OCR (grayscale preprocessing)', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_counter += 1

cap.release()
cv2.destroyAllWindows()