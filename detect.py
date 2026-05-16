"""
Детектор ценников на видео.

Зоны ценника:
  БЕЛАЯ ЗОНА (верх):
    ├─ Левые 55%  → product_name  (тёмный текст на белом)
    ├─ Правые 45% → QR-код        (декодируется pyzbar)
    │    ├─ верх QR              → сам QR
    │    └─ низ  (~30%)          → price_default (цена без карты, напр. 368)
  КРАСНАЯ ЗОНА (низ):
    ├─ Левые 28%  → discount_amount + мелкий текст (дата, id_sku)
    ├─ 28–78%    → price_card    (белый текст, крупные цифры)
    └─ Правые 22% → штрихкод     (pyzbar + OCR цифр под ним)

QR содержит данные через разделитель — парсим price1_qr…price4_qr.
Дедупликация: трекинг bbox по IOU, запись при уходе ценника из кадра.
"""

import cv2, csv, re, os, time
import numpy as np
import pytesseract
from ultralytics import YOLO
from dataclasses import dataclass, field

try:
    from pyzbar import pyzbar as _pyzbar
    def decode_codes(img): return [c.data.decode("utf-8","ignore") for c in _pyzbar.decode(img)]
except ImportError:
    print("⚠ pyzbar не установлен — QR/штрихкод не будет декодирован")
    def decode_codes(img): return []

# ══════════════════════════ НАСТРОЙКИ ═════════════════════════════════════════
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

VIDEO_PATH  = "test.mp4"
OUTPUT_CSV  = "results.csv"
YOLO_MODEL  = "runs/detect/train-2/weights/best.pt"
CONF_THRESH = 0.5
FRAME_SKIP  = 5
ROTATION    = cv2.ROTATE_90_COUNTERCLOCKWISE   # 270° по часовой

UPSCALE_WHITE   = 2.0
UPSCALE_PRICE   = 2.0
UPSCALE_DISC    = 5.0
UPSCALE_BARCODE = 8.0
UPSCALE_QR      = 4.0
UPSCALE_SMALL   = 7.0   # для мелкого текста (дата, id_sku)

IOU_THRESHOLD  = 0.45
CONFIRM_FRAMES = 10
MAX_MISS       = 30
# ══════════════════════════════════════════════════════════════════════════════

CSV_FIELDS = [
    "filename","product_name","price_default","price_card","price_discount",
    "barcode","discount_amount","id_sku","print_datetime","code",
    "additional_info","color","special_symbols","frame_timestamp",
    "x_min","y_min","x_max","y_max",
    "qr_code_barcode","price1_qr","price2_qr","price3_qr","price4_qr",
    "wholesale_level_1_count","wholesale_level_1_price",
    "wholesale_level_2_count","wholesale_level_2_price",
    "action_price_qr","action_code_qr",
]

# ──────────────────────── УТИЛИТЫ ─────────────────────────────────────────────

def _up(img, factor):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w*factor), int(h*factor)), interpolation=cv2.INTER_CUBIC)

def _pad(b):
    return cv2.copyMakeBorder(b, 10,10,10,10, cv2.BORDER_CONSTANT, value=255)

def _otsu_black_on_white(img_bgr, scale):
    up   = _up(img_bgr, scale)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(b) < 127: b = cv2.bitwise_not(b)
    return b

def _clahe_adap(img_bgr, scale, block=15, c=6):
    """CLAHE + адаптивный порог — лучше для мелкого текста на цветном фоне."""
    up   = _up(img_bgr, scale)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    cl   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
    gray = cl.apply(gray)
    b    = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, block, c)
    if np.mean(b) < 127: b = cv2.bitwise_not(b)
    return b

def _red_zone_white_text(img_bgr, scale):
    """R - max(G,B): выделяет белый текст на красном фоне."""
    up = _up(img_bgr, scale)
    b, g, r = cv2.split(up.astype(np.int16))
    sig  = np.clip(r - np.maximum(g, b), 0, 255).astype(np.uint8)
    inv  = cv2.bitwise_not(sig)
    _, b = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k2   = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    b    = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k2, iterations=1)
    return cv2.bitwise_not(b)

def _trim_red_edges(img):
    """Обрезает чёрные скруглённые углы по HSV красной маске."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m   = cv2.bitwise_or(cv2.inRange(hsv,(0,80,80),(10,255,255)),
                         cv2.inRange(hsv,(160,80,80),(180,255,255)))
    col = m.sum(axis=0)/(img.shape[0]*255)
    row = m.sum(axis=1)/(img.shape[1]*255)
    cs  = next((i for i,v in enumerate(col) if v>0.2), 0)
    ce  = next((i for i,v in enumerate(reversed(col)) if v>0.2), 0)
    rs  = next((i for i,v in enumerate(row) if v>0.2), 0)
    re_ = next((i for i,v in enumerate(reversed(row)) if v>0.2), 0)
    h,w = img.shape[:2]
    return img[rs: h-re_ if re_ else h, cs: w-ce if ce else w]

def _split_zones(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m   = cv2.bitwise_or(cv2.inRange(hsv,(0,80,80),(10,255,255)),
                         cv2.inRange(hsv,(160,80,80),(180,255,255)))
    ratio = m.sum(axis=1)/(img.shape[1]*255)
    split = next((i for i,r in enumerate(ratio) if r>0.30), None)
    if split is None: return img, None
    if split < 5:     return None, img
    return img[:split], img[split:]

# ──────────────────────── OCR ─────────────────────────────────────────────────

def _ocr(img_proc, cfg, min_h=60):
    if img_proc.shape[0] < min_h:
        s = min_h/img_proc.shape[0]
        img_proc = cv2.resize(img_proc, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    data = pytesseract.image_to_data(img_proc, lang="rus+eng", config=cfg,
                                     output_type=pytesseract.Output.DICT)
    words = [(t.strip(), c) for t,c in zip(data["text"],data["conf"])
             if t.strip() and c > 0]
    text = " ".join(w for w,_ in words)
    conf = float(np.mean([c for _,c in words])) if words else 0.0
    return text, conf

CFG_BLOCK = "--oem 3 --psm 6"
CFG_LINE  = "--oem 3 --psm 7"
CFG_DIGIT = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789."

# ──────────────────────── ПАРСИНГ ─────────────────────────────────────────────

def _price(text):
    m = re.search(r"\d[\d\s]*[.,]\d{2}", text)
    if m: return m.group().replace(" ","").replace(",",".")
    m = re.search(r"\d{2,}", text)
    return m.group() if m else ""

def _discount(text):
    m = re.search(r"-?\d+\s*%", text)
    return m.group().replace(" ","") if m else ""

def _digits_only(text):
    return re.sub(r"[^\d]","", text)

def _parse_qr(raw: str) -> dict:
    """
    QR на ценниках Ленты содержит данные через разделители.
    Формат примерно: barcode|price1|price2||price_card|...
    Пробуем несколько разделителей.
    """
    result = {"qr_code_barcode": raw, "price1_qr":"","price2_qr":"",
              "price3_qr":"","price4_qr":"","action_price_qr":"нет",
              "action_code_qr":"нет"}
    # Ищем числовые поля — цены и штрихкод
    parts = re.split(r"[|;\t]", raw)
    prices = []
    barcode = ""
    for p in parts:
        p = p.strip()
        if re.fullmatch(r"\d{8,13}", p):
            barcode = p
        elif re.fullmatch(r"\d+[.,]\d{2}", p):
            prices.append(p.replace(",","."))
    if barcode:
        result["qr_code_barcode"] = barcode
    for i, pr in enumerate(prices[:4]):
        result[f"price{i+1}_qr"] = pr
    return result

# ──────────────────────── ОБРАБОТКА КРОПА ─────────────────────────────────────

def process_crop(crop: np.ndarray) -> dict:
    r = {f:"" for f in CSV_FIELDS}
    r.update({"code":"нет","additional_info":"нет","color":"red",
               "special_symbols":"нет","wholesale_level_1_count":"нет",
               "wholesale_level_1_price":"нет","wholesale_level_2_count":"нет",
               "wholesale_level_2_price":"нет","action_price_qr":"нет",
               "action_code_qr":"нет"})
    confs = []

    white, red = _split_zones(crop)

    # ── БЕЛАЯ ЗОНА ──────────────────────────────────────────────────────────
    if white is not None and white.shape[0] >= 10:
        wh, ww = white.shape[:2]

        # Название товара — левые 55%
        name_z = white[:, :int(ww*0.55)]
        if name_z.shape[1] > 10:
            proc = _otsu_black_on_white(name_z, UPSCALE_WHITE)
            text, conf = _ocr(_pad(proc), CFG_BLOCK)
            r["product_name"] = " ".join(text.split())
            confs.append(conf)

        # QR зона — правые 45%
        qr_z = white[:, int(ww*0.55):]
        if qr_z.shape[1] > 10:
            qzh, qzw = qr_z.shape[:2]

            # QR-код — верхние 70%
            qr_img = qr_z[:int(qzh*0.70), :]
            proc_qr = _otsu_black_on_white(qr_img, UPSCALE_QR)
            codes = decode_codes(proc_qr)
            if not codes:
                # Попробовать оригинал без обработки
                codes = decode_codes(_up(qr_img, UPSCALE_QR))
            if codes:
                parsed = _parse_qr(codes[0])
                r.update(parsed)

            # price_default — нижние 30% правой зоны (цена без карты, напр. "368")
            pd_z = qr_z[int(qzh*0.70):, :]
            if pd_z.shape[0] > 4:
                proc_pd = _otsu_black_on_white(pd_z, UPSCALE_SMALL)
                text, conf = _ocr(_pad(proc_pd), CFG_DIGIT)
                val = _price(text)
                if val:
                    r["price_default"] = val
                    confs.append(conf)

    # ── КРАСНАЯ ЗОНА ────────────────────────────────────────────────────────
    if red is not None and red.shape[0] >= 10:
        img = red[2:-2, 2:-2] if red.shape[0]>6 and red.shape[1]>6 else red
        rh, rw = img.shape[:2]

        # Скидка — левые 28%
        disc_z = img[:, :int(rw*0.28)]
        if disc_z.shape[1] > 10:
            disc_t = _trim_red_edges(disc_z)
            dh, dw = disc_t.shape[:2]

            # Крупный текст скидки — верхние 42% (сам процент)
            disc_top = disc_t[:int(dh*0.42), :]
            if disc_top.shape[0] > 4:
                proc = _otsu_black_on_white(disc_top, UPSCALE_DISC)
                text, conf = _ocr(_pad(proc), CFG_LINE)
                val = _discount(text)
                if val:
                    r["discount_amount"] = val
                    confs.append(conf)

            # Мелкий текст под скидкой — нижние 58% (дата печати, id_sku)
            disc_bot = disc_t[int(dh*0.42):, :]
            if disc_bot.shape[0] > 4:
                proc = _clahe_adap(disc_bot, UPSCALE_SMALL, block=13, c=5)
                text, conf = _ocr(_pad(proc), CFG_BLOCK)
                # id_sku — длинное число
                m_id = re.search(r"\d{10,}", text)
                if m_id: r["id_sku"] = m_id.group()
                # Дата — ищем ДД.ММ.ГГГГ или похожее
                m_dt = re.search(r"\d{1,2}[./]\d{1,2}[./]\d{2,4}", text)
                if m_dt: r["print_datetime"] = m_dt.group()
                if conf > 0: confs.append(conf)

        # Цена по карте — средние 28–78% (крупные белые цифры)
        price_z = img[:, int(rw*0.28):int(rw*0.78)]
        if price_z.shape[1] > 10:
            proc = _red_zone_white_text(price_z, UPSCALE_PRICE)
            text, conf = _ocr(_pad(proc), CFG_LINE)
            val = _price(text)
            if val:
                r["price_card"] = val
                confs.append(conf)

        # Штрихкод — правые 22%
        bar_z = img[:, int(rw*0.78):]
        if bar_z.shape[1] > 8:
            # 1) pyzbar на разных апскейлах
            bar_found = ""
            for sc in [UPSCALE_BARCODE, 6.0, 10.0]:
                proc_b = _otsu_black_on_white(bar_z, sc)
                codes  = decode_codes(proc_b)
                if not codes:
                    codes = decode_codes(_up(bar_z, sc))  # без бинаризации
                if codes:
                    bar_found = codes[0]
                    break

            # 2) Если pyzbar не взял — OCR цифр под штрихкодом
            if not bar_found:
                bzh = bar_z.shape[0]
                digits_z = bar_z[int(bzh*0.65):, :]   # нижние 35% — цифры EAN
                if digits_z.shape[0] > 3:
                    proc_d = _clahe_adap(digits_z, UPSCALE_SMALL, block=11, c=4)
                    text, _ = _ocr(_pad(proc_d), CFG_DIGIT)
                    d = _digits_only(text)
                    if len(d) >= 8:
                        bar_found = d

            if bar_found:
                r["barcode"] = bar_found
                if not r.get("qr_code_barcode"):
                    r["qr_code_barcode"] = bar_found

    r["_confidence"] = float(np.mean(confs)) if confs else 0.0
    return r

# ──────────────────────── ТРЕКЕР ──────────────────────────────────────────────

@dataclass
class Track:
    bbox:         tuple  = (0,0,0,0)
    detect_count: int    = 0
    miss_count:   int    = 0
    saved:        bool   = False
    best_conf:    float  = 0.0
    data:         dict   = field(default_factory=dict)

def _iou(a, b):
    ix1,iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2,iy2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    if not inter: return 0.0
    aa = (a[2]-a[0])*(a[3]-a[1]); ab = (b[2]-b[0])*(b[3]-b[1])
    return inter/(aa+ab-inter)

class TagTracker:
    def __init__(self): self._tracks: list[Track] = []

    def update(self, detections, frame_idx, fps, video_name):
        matched_t = set(); matched_d = set()
        for di,(bbox,ocr) in enumerate(detections):
            best_iou, best_ti = 0.0, -1
            for ti,t in enumerate(self._tracks):
                v = _iou(bbox, t.bbox)
                if v > best_iou: best_iou,best_ti = v,ti
            if best_iou >= IOU_THRESHOLD:
                matched_t.add(best_ti); matched_d.add(di)
                t = self._tracks[best_ti]
                t.bbox = bbox; t.detect_count += 1; t.miss_count = 0
                new_conf = ocr.pop("_confidence", 0.0)
                if new_conf > t.best_conf:
                    t.best_conf = new_conf
                    # Обновляем только непустые поля
                    for k,v in ocr.items():
                        if v and v != "нет": t.data[k] = v
            else:
                d = dict(ocr)
                conf = d.pop("_confidence", 0.0)
                d["filename"] = video_name
                d["frame_timestamp"] = f"{frame_idx/fps:.2f}s"
                d["x_min"],d["y_min"],d["x_max"],d["y_max"] = bbox
                t = Track(bbox=bbox, detect_count=1, best_conf=conf, data=d)
                self._tracks.append(t)

        for ti,t in enumerate(self._tracks):
            if ti not in matched_t: t.miss_count += 1

        to_save, alive = [], []
        for t in self._tracks:
            if t.miss_count > MAX_MISS:
                if t.detect_count >= CONFIRM_FRAMES and not t.saved:
                    to_save.append(t); t.saved = True
            else:
                alive.append(t)
        self._tracks = alive
        return to_save

    def flush(self):
        result = []
        for t in self._tracks:
            if t.detect_count >= CONFIRM_FRAMES and not t.saved:
                result.append(t); t.saved = True
        return result

# ──────────────────────── MAIN ────────────────────────────────────────────────

def _row(track: Track, video_name: str) -> dict:
    row = {f:"" for f in CSV_FIELDS}
    row.update({"code":"нет","additional_info":"нет","color":"red",
                "special_symbols":"нет","wholesale_level_1_count":"нет",
                "wholesale_level_1_price":"нет","wholesale_level_2_count":"нет",
                "wholesale_level_2_price":"нет","action_price_qr":"нет",
                "action_code_qr":"нет"})
    row.update(track.data)
    row["filename"] = video_name
    return row

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Не удалось открыть {VIDEO_PATH}"); return

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25
    model      = YOLO(YOLO_MODEL)
    tracker    = TagTracker()
    video_name = os.path.basename(VIDEO_PATH)
    frame_idx  = 0

    out = open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(out, fieldnames=CSV_FIELDS)
    writer.writeheader()

    print(f"▶ {VIDEO_PATH}  FPS={fps:.1f}  skip={FRAME_SKIP}")
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        frame = cv2.rotate(frame, ROTATION)

        if frame_idx % FRAME_SKIP != 0: continue

        results = model(frame, conf=CONF_THRESH, verbose=False)
        detections = []
        fh, fw = frame.shape[:2]

        for box in results[0].boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            x1=max(0,x1-5); y1=max(0,y1-5)
            x2=min(fw,x2+5); y2=min(fh,y2+5)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue

            ocr = process_crop(crop)
            ocr["x_min"]=x1; ocr["y_min"]=y1
            ocr["x_max"]=x2; ocr["y_max"]=y2
            ocr["frame_timestamp"] = f"{frame_idx/fps:.2f}s"
            detections.append(((x1,y1,x2,y2), ocr))

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,200,0),2)
            lbl = ocr.get("price_card") or ocr.get("price_default") or "?"
            cv2.putText(frame, lbl, (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,0), 2)

        ready = tracker.update(detections, frame_idx, fps, video_name)
        for t in ready:
            row = _row(t, video_name)
            writer.writerow(row)
            print(f"  ✅ {row['product_name'][:40] or '?'} | "
                  f"цена={row['price_card']} | без карты={row['price_default']} | "
                  f"скидка={row['discount_amount']} | barcode={row['barcode']} | "
                  f"QR={row['qr_code_barcode'][:20] if row['qr_code_barcode'] else ''}")

        cv2.imshow("Price Tag Detector",
                   cv2.resize(frame, None, fx=0.6, fy=0.6))
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    for t in tracker.flush():
        row = _row(t, video_name)
        writer.writerow(row)

    elapsed = time.time()-t0
    out.close(); cap.release(); cv2.destroyAllWindows()
    print(f"\n✔ Готово за {elapsed:.1f}с → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()