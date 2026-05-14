from ultralytics import YOLO
import cv2

model = YOLO('runs/detect/train-2/weights/best.pt')   # ваш файл
cap = cv2.VideoCapture('test.mp4')

# Коэффициент уменьшения для отображения (например, 0.25 = 1/4 от оригинала)
DISPLAY_SCALE = 0.25

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Поворот на 270° по часовой стрелке
    rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)  # теперь размер (2160, 3840)

    # 2. Детекция на полном повёрнутом кадре
    results = model(rotated, stream=True)

    for result in results:
        annotated = result.plot()   # аннотированный кадр в полном размере (2160×3840)

    # 3. Масштабируем для отображения, чтобы окно не уходило за экран
    h, w = annotated.shape[:2]
    new_w = int(w * DISPLAY_SCALE)
    new_h = int(h * DISPLAY_SCALE)
    display_frame = cv2.resize(annotated, (new_w, new_h), interpolation=cv2.INTER_AREA)

    cv2.imshow('YOLO + поворот (масштабированный показ)', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()