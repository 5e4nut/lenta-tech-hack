#!/usr/bin/env python3
"""
video_detect.py — детекция ценников на видео с трекингом и дедупликацией.

Пайплайн
────────
  mp4-файл
    └─► undistort (fisheye коррекция)
    └─► поворот на +90° (компенсация монтажа камеры)
    └─► YOLO detect  (раз в DETECT_EVERY_N кадров, по умолчанию ~1 сек)
    └─► ByteTrack     (каждый кадр, получает bbox от YOLO или из кэша)
    └─► фильтр качества: bbox полностью в кадре + conf ≥ MIN_CONF
    └─► дедупликация:  track_id → обрабатываем один раз,
                       повтор только если прошло > REPROCESS_AFTER_SEC секунд
    └─► process_numpy_frame() из det.py → строка CSV
    └─► preview-окно с bbox и track_id

Использование
─────────────
  python video_detect.py input.mp4
  python video_detect.py input.mp4 -o tags.csv
  python video_detect.py input.mp4 --model best.pt --ollama-model qwen2.5vl:3b-q4_K_M
  python video_detect.py input.mp4 --no-undistort   # отключить fisheye коррекцию
  python video_detect.py input.mp4 --no-preview     # без окна

Зависимости
───────────
  pip install ultralytics opencv-python numpy
  (det.py и его зависимости должны быть в той же папке)
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── импортируем det.py напрямую ───────────────────────────────────────────────
try:
    import det as det_module
except ImportError as e:
    print(f"[FATAL] Не могу импортировать det.py: {e}")
    print("        Убедитесь что det.py находится в той же папке что и video_detect.py")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("[FATAL] ultralytics не установлен: pip install ultralytics")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  ПАРАМЕТРЫ КАМЕРЫ  (из example_undistort.py)
# ══════════════════════════════════════════════════════════════════════════════

CAM_IMAGE_SIZE   = (3840, 2160)    # ширина × высота пикселей
CAM_DIAGONAL_MM  = 16.0 / 2.8     # ≈ 5.714 мм (vidicon пересчёт)
CAM_FOCAL_MM     = 2.8            # фокусное расстояние мм
CAM_DIST_COEFFS  = [-0.276, 0.06, 0.0084, -0.0016, -0.0044]  # k1 k2 p1 p2 k3

# ── тюнинг ────────────────────────────────────────────────────────────────────
DETECT_EVERY_N_SEC  = 1.0      # как часто запускать YOLO (в секундах видео)
MIN_CONF            = 0.55     # минимальная уверенность детекции
MARGIN_FRAC         = 0.05     # bbox должен быть дальше этой доли от края кадра
REPROCESS_AFTER_SEC = 60.0     # через сколько секунд повторно обрабатывать тот же track_id
TRACK_LOST_SEC      = 3.0      # через сколько секунд без детекции track считать потерянным
YOLO_IOU            = 0.5      # IoU для трекера
YOLO_CONF           = 0.35     # conf-порог внутри YOLO (ниже чем MIN_CONF — первичный фильтр)
PREVIEW_SCALE       = 0.35     # масштаб preview-окна
PREVIEW_WIN         = "PriceTags — press Q to quit"


# ══════════════════════════════════════════════════════════════════════════════
#  КОРРЕКЦИЯ ДИСТОРСИИ
# ══════════════════════════════════════════════════════════════════════════════

class UndistortProcessor:
    """
    Вычисляет карты коррекции один раз, затем применяет cv2.remap() к каждому кадру.
    alpha=0  — максимальный кроп, все пиксели валидны (потери ~10-15% по краям).
    alpha=1  — сохранить всё поле зрения, чёрные углы.
    """

    def __init__(self, image_size=CAM_IMAGE_SIZE, diagonal_mm=CAM_DIAGONAL_MM,
                 focal_mm=CAM_FOCAL_MM, dist_coeffs=CAM_DIST_COEFFS, alpha=0):
        self.w, self.h = image_size
        self.dist = np.array(dist_coeffs, dtype=np.float32)
        self.K = self._build_camera_matrix(diagonal_mm, focal_mm)
        self.map1, self.map2, self.roi = self._build_maps(alpha)
        print(f"[Undistort] K matrix:\n{self.K}")
        print(f"[Undistort] ROI after crop: {self.roi}")

    def _build_camera_matrix(self, diagonal_mm: float, focal_mm: float) -> np.ndarray:
        aspect = self.w / self.h
        h_mm = diagonal_mm / math.sqrt(aspect ** 2 + 1)
        w_mm = aspect * h_mm
        fx = focal_mm * self.w / w_mm
        fy = focal_mm * self.h / h_mm
        return np.array([
            [fx, 0,  self.w / 2],
            [0,  fy, self.h / 2],
            [0,  0,  1],
        ], dtype=np.float32)

    def _build_maps(self, alpha: float):
        new_K, roi = cv2.getOptimalNewCameraMatrix(
            self.K, self.dist, (self.w, self.h), alpha, (self.w, self.h)
        )
        map1, map2 = cv2.initUndistortRectifyMap(
            self.K, self.dist, None, new_K, (self.w, self.h), cv2.CV_32FC1
        )
        return map1, map2, roi

    def process(self, frame: np.ndarray) -> np.ndarray:
        undist = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
        x, y, w, h = self.roi
        return undist[y:y + h, x:x + w]


# ══════════════════════════════════════════════════════════════════════════════
#  МЕНЕДЖЕР ДЕДУПЛИКАЦИИ
# ══════════════════════════════════════════════════════════════════════════════

class DeduplicationManager:
    """
    Хранит per-track_id статус:
      - когда последний раз обрабатывался (video-время в секундах)
      - когда последний раз видели в кадре
    Трек считается "новым" если:
      1. Никогда не обрабатывался, ИЛИ
      2. Прошло > REPROCESS_AFTER_SEC с момента последней обработки
    """

    def __init__(self, reprocess_after: float = REPROCESS_AFTER_SEC,
                 track_lost_after: float = TRACK_LOST_SEC):
        self.reprocess_after = reprocess_after
        self.track_lost_after = track_lost_after
        self._last_processed: dict[int, float] = {}   # track_id → video_time
        self._last_seen: dict[int, float] = {}         # track_id → video_time

    def mark_seen(self, track_id: int, video_time: float):
        self._last_seen[track_id] = video_time

    def should_process(self, track_id: int, video_time: float) -> bool:
        last = self._last_processed.get(track_id)
        if last is None:
            return True
        return (video_time - last) > self.reprocess_after

    def mark_processed(self, track_id: int, video_time: float):
        self._last_processed[track_id] = video_time

    def cleanup(self, current_time: float):
        """Удалить треки которые давно не видели (освободить память)."""
        stale = [tid for tid, t in self._last_seen.items()
                 if (current_time - t) > self.track_lost_after * 10]
        for tid in stale:
            self._last_processed.pop(tid, None)
            self._last_seen.pop(tid, None)

    def stats(self) -> str:
        return f"known_tracks={len(self._last_processed)}"


# ══════════════════════════════════════════════════════════════════════════════
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ══════════════════════════════════════════════════════════════════════════════

def rotate_90cw(frame: np.ndarray) -> np.ndarray:
    """Повернуть кадр на 90° по часовой стрелке (компенсация монтажа камеры)."""
    return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)


def bbox_fully_visible(x1: int, y1: int, x2: int, y2: int,
                       frame_w: int, frame_h: int,
                       margin: float = MARGIN_FRAC) -> bool:
    """Проверить что bbox полностью внутри кадра с отступом margin."""
    mx = int(frame_w * margin)
    my = int(frame_h * margin)
    return x1 >= mx and y1 >= my and x2 <= (frame_w - mx) and y2 <= (frame_h - my)


def crop_with_padding(frame: np.ndarray, x1: int, y1: int,
                      x2: int, y2: int, pad: int = 8) -> np.ndarray:
    """Вырезать bbox с небольшим паддингом по краям."""
    h, w = frame.shape[:2]
    x1c = max(0, x1 - pad)
    y1c = max(0, y1 - pad)
    x2c = min(w, x2 + pad)
    y2c = min(h, y2 + pad)
    return frame[y1c:y2c, x1c:x2c]


def draw_overlay(frame: np.ndarray, tracks: list,
                 processed_ids: set, pending_ids: set) -> np.ndarray:
    """
    Нарисовать bbox и track_id поверх кадра для preview.
    Цвета:
      зелёный  — уже обработан в этот раз
      жёлтый   — в очереди на обработку
      голубой  — обнаружен, ждёт подходящего кадра
    """
    out = frame.copy()
    for (tid, x1, y1, x2, y2, conf) in tracks:
        if tid in processed_ids:
            color = (0, 200, 0)
            label = f"#{tid} done"
        elif tid in pending_ids:
            color = (0, 200, 255)
            label = f"#{tid} queued"
        else:
            color = (200, 200, 0)
            label = f"#{tid} {conf:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(y1 - 6, th + 4)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
        cv2.putText(out, label, (x1 + 2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  ОСНОВНОЙ ЦИКЛ
# ══════════════════════════════════════════════════════════════════════════════

def run(args):
    # ── инициализация Ollama / det ────────────────────────────────────────────
    print(f"\n[Init] Подключение к Ollama {args.ollama_host} ...")
    resolved_model = det_module.init_det_module(args.ollama_host, args.ollama_model)
    print(f"[Init] Модель Ollama: {resolved_model}")

    # ── YOLO ──────────────────────────────────────────────────────────────────
    print(f"[Init] Загрузка YOLO: {args.model}")
    yolo = YOLO(args.model)
    print(f"[Init] YOLO загружена ✓")

    # ── undistort ─────────────────────────────────────────────────────────────
    undist = None
    if not args.no_undistort:
        print("[Init] Вычисление карт коррекции дисторсии ...")
        undist = UndistortProcessor()
    else:
        print("[Init] Коррекция дисторсии отключена (--no-undistort)")

    # ── видео ─────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        print(f"[FATAL] Не могу открыть видео: {args.input}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detect_every_n = max(1, int(fps * DETECT_EVERY_N_SEC))
    print(f"[Video] {args.input.name}  |  {total_frames} кадров  |  {fps:.1f} FPS")
    print(f"[Video] YOLO каждые {detect_every_n} кадров (~{DETECT_EVERY_N_SEC}с)")
    print(f"[Video] Выходной CSV: {args.output}\n")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_file = open(args.output, "w", newline="", encoding="utf-8-sig")
    csv_writer = csv.DictWriter(csv_file, fieldnames=det_module.CSV_COLUMNS,
                                extrasaction="ignore")
    csv_writer.writeheader()
    csv_written = 0

    # ── состояние ─────────────────────────────────────────────────────────────
    dedup = DeduplicationManager(
        reprocess_after=args.reprocess_after,
        track_lost_after=TRACK_LOST_SEC,
    )
    # track_id → последний bbox (для кадров между YOLO-детектами)
    last_boxes: dict[int, tuple] = {}
    # track_id-ы обработанных за текущую "сессию" (для цвета overlay)
    session_processed: set = set()
    # track_id-ы в очереди прямо сейчас
    pending_ids: set = set()

    frame_idx = 0
    last_yolo_result = []   # [(track_id, x1, y1, x2, y2, conf), ...]

    t_start = time.time()
    print("[Run] Обработка видео... (нажмите Q в окне preview для выхода)\n")

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        video_time = frame_idx / fps   # секунды с начала видео

        # ── предобработка ────────────────────────────────────────────────────
        frame = raw_frame

        if undist is not None:
            frame = undist.process(frame)

        # Поворот: видео снято камерой повёрнутой против часовой стрелки,
        # компенсируем вращением кадра на 90° по часовой стрелке.
        frame = rotate_90cw(frame)

        fh, fw = frame.shape[:2]

        # ── YOLO detect (раз в N кадров) ─────────────────────────────────────
        if frame_idx % detect_every_n == 0:
            results = yolo.track(
                frame,
                persist=True,       # ByteTrack хранит состояние между вызовами
                tracker="bytetrack.yaml",
                conf=YOLO_CONF,
                iou=YOLO_IOU,
                verbose=False,
            )

            new_boxes = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                ids   = boxes.id      # может быть None если трекер не нашёл
                xywh  = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                if ids is not None:
                    for tid, xyxy, conf in zip(ids.cpu().numpy(), xywh, confs):
                        tid = int(tid)
                        x1, y1, x2, y2 = map(int, xyxy)
                        new_boxes.append((tid, x1, y1, x2, y2, float(conf)))
                        dedup.mark_seen(tid, video_time)

            last_yolo_result = new_boxes
            # Обновить кэш боксов
            seen_this_frame = {b[0] for b in new_boxes}
            last_boxes = {tid: box for tid, box in last_boxes.items()
                          if tid in seen_this_frame}
            for b in new_boxes:
                last_boxes[b[0]] = b

        # ── для кадров между детектами — используем последние известные боксы ─
        current_tracks = list(last_boxes.values())

        # ── фильтрация и постановка в очередь на обработку ───────────────────
        to_process = []
        for (tid, x1, y1, x2, y2, conf) in current_tracks:
            if conf < MIN_CONF:
                continue
            if not bbox_fully_visible(x1, y1, x2, y2, fw, fh):
                continue
            if not dedup.should_process(tid, video_time):
                continue
            # Попал в кадр и прошёл все фильтры — берём на обработку
            to_process.append((tid, x1, y1, x2, y2, conf))

        # ── отправка в Ollama ─────────────────────────────────────────────────
        for (tid, x1, y1, x2, y2, conf) in to_process:
            dedup.mark_processed(tid, video_time)   # сразу помечаем чтобы не дублировать
            pending_ids.add(tid)

            crop = crop_with_padding(frame, x1, y1, x2, y2)
            fname = f"frame_{frame_idx:06d}_track{tid}"

            print(f"  [{frame_idx:6d} | {video_time:7.1f}s] "
                  f"Обрабатываю track#{tid}  bbox=({x1},{y1},{x2},{y2})  conf={conf:.2f}",
                  flush=True)
            t0 = time.time()
            try:
                row = det_module.process_numpy_frame(
                    img=crop,
                    frame_filename=fname,
                    host=args.ollama_host,
                    model=resolved_model,
                    frame_timestamp=video_time,
                    bbox=(x1, y1, x2, y2),
                )
                csv_writer.writerow(row)
                csv_file.flush()
                csv_written += 1
                session_processed.add(tid)
                print(f"    → {row.get('tag_type','?')}  '{row.get('product_name','')[:50]}'  "
                      f"({time.time()-t0:.1f}s)  [CSV строк: {csv_written}]")
            except Exception as e:
                print(f"    ✗ Ошибка обработки track#{tid}: {e}")
            finally:
                pending_ids.discard(tid)

        # ── периодическая чистка памяти ───────────────────────────────────────
        if frame_idx % (detect_every_n * 60) == 0:
            dedup.cleanup(video_time)

        # ── preview ───────────────────────────────────────────────────────────
        if not args.no_preview:
            vis = draw_overlay(frame, current_tracks, session_processed, pending_ids)

            # Статус-строка
            elapsed = time.time() - t_start
            status = (f"Frame {frame_idx}/{total_frames}  |  "
                      f"{video_time:.1f}s  |  "
                      f"CSV: {csv_written}  |  "
                      f"{dedup.stats()}")
            cv2.putText(vis, status, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, status, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

            small = cv2.resize(vis, (int(fw * PREVIEW_SCALE), int(fh * PREVIEW_SCALE)))
            cv2.imshow(PREVIEW_WIN, small)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                print("\n[Run] Выход по нажатию Q")
                break

        # ── прогресс в консоли ────────────────────────────────────────────────
        if frame_idx % (detect_every_n * 10) == 0:
            pct = frame_idx / total_frames * 100 if total_frames else 0
            elapsed = time.time() - t_start
            print(f"  Прогресс: {pct:.1f}%  кадр {frame_idx}/{total_frames}  "
                  f"время {elapsed:.0f}с  CSV строк: {csv_written}  {dedup.stats()}")

    # ── завершение ────────────────────────────────────────────────────────────
    cap.release()
    csv_file.close()
    if not args.no_preview:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\n✅ Готово за {elapsed:.0f}с")
    print(f"   Обработано кадров: {frame_idx}")
    print(f"   Записано в CSV:    {csv_written} строк → {args.output}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Детекция ценников на видео (YOLO + ByteTrack + Ollama)"
    )
    p.add_argument("input", type=Path,
                   help="Путь к .mp4 / .avi видеофайлу")
    p.add_argument("-o", "--output", type=Path, default=Path("video_tags.csv"),
                   help="Выходной CSV (по умолчанию: video_tags.csv)")
    p.add_argument("--model", default="best.pt",
                   help="Путь к YOLO-модели (по умолчанию: best.pt)")
    p.add_argument("--ollama-model", default=det_module.DEFAULT_MODEL,
                   help=f"Модель Ollama (по умолчанию: {det_module.DEFAULT_MODEL})")
    p.add_argument("--ollama-host", default=det_module.DEFAULT_HOST,
                   help=f"Адрес Ollama (по умолчанию: {det_module.DEFAULT_HOST})")
    p.add_argument("--reprocess-after", type=float, default=REPROCESS_AFTER_SEC,
                   help=f"Секунд до повторной обработки трека (по умолчанию: {REPROCESS_AFTER_SEC})")
    p.add_argument("--no-undistort", action="store_true",
                   help="Отключить fisheye-коррекцию")
    p.add_argument("--no-preview", action="store_true",
                   help="Не показывать preview-окно")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.input.exists():
        print(f"[FATAL] Файл не найден: {args.input}")
        sys.exit(1)

    run(args)