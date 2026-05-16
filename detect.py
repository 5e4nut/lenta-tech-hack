#!/usr/bin/env python3
"""
Парсер ценников → CSV
Текст распознаётся локальной нейросетью через Ollama (Qwen2.5-VL).
QR-коды и штрих-коды читает pyzbar (без сети, без ключей).

═══════════════════════════════════════════════════════
 УСТАНОВКА (один раз)
═══════════════════════════════════════════════════════

1. Ollama:
   Linux/macOS:  curl -fsSL https://ollama.ai/install.sh | sh
   Windows:      https://ollama.ai/download

2. Скачать модель:
   ollama pull qwen2.5vl:7b        # ~5 GB — рекомендуется
   ollama pull qwen2.5vl:3b        # ~2.5 GB — если мало RAM/VRAM

3. Python-зависимости:
   pip install opencv-python-headless pillow pyzbar numpy requests

4. pyzbar системная либа:
   Ubuntu/Debian:  sudo apt install libzbar0
   macOS:          brew install zbar
   Windows:        pip install pyzbar  (dll идёт в комплекте)

═══════════════════════════════════════════════════════
 ИСПОЛЬЗОВАНИЕ
═══════════════════════════════════════════════════════
   # Ollama должна быть запущена: ollama serve
   python detect.py ./папка_с_ценниками
   python detect.py ./папка -o result.csv
   python detect.py img1.jpg img2.png -o result.csv

   # Другая модель:
   python detect.py ./папка --model qwen2.5vl:3b

   # Другой адрес Ollama (если не localhost):
   python detect.py ./папка --host http://192.168.1.10:11434
"""

import os
import re
import csv
import sys
import json
import base64
import argparse
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    HAS_PYZBAR = True
except ImportError:
    HAS_PYZBAR = False
    print("[!] pyzbar не найден — QR и штрих-коды не будут читаться.")
    print("    pip install pyzbar   |   Ubuntu: sudo apt install libzbar0\n")

# ── Настройки ────────────────────────────────────────────────────────────────
# ИСПРАВЛЕНО: заменили llava:7b → qwen2.5vl:7b (намного лучше читает русский текст и OCR)
DEFAULT_MODEL = "qwen2.5vl:3b-q4_K_M"
DEFAULT_HOST  = "http://localhost:11434"

# ── Колонки CSV ───────────────────────────────────────────────────────────────
CSV_COLUMNS = [
    "filename", "product_name", "price_default", "price_card", "price_discount",
    "barcode", "discount_amount", "id_sku", "print_datetime", "code",
    "additional_info", "color", "special_symbols", "frame_timestamp",
    "x_min", "y_min", "x_max", "y_max",
    "qr_code_barcode", "price1_qr", "price2_qr", "price3_qr", "price4_qr",
    "wholesale_level_1_count", "wholesale_level_1_price",
    "wholesale_level_2_count", "wholesale_level_2_price",
    "action_price_qr", "action_code_qr",
]

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

# ── Промпт для модели ─────────────────────────────────────────────────────────
PROMPT = """Ты — OCR-система для российских магазинных ценников. Внимательно прочитай все надписи на ценнике и верни строго JSON-объект.

Извлеки следующие поля:
{
  "product_name": "полное название товара с весом/объёмом и страной в скобках",
  "price_default": "цена без карты лояльности — число с точкой, например 72.59",
  "price_card": "цена по карте лояльности — число с точкой, например 55.39",
  "price_discount": "акционная или оптовая цена если есть — число с точкой",
  "discount_amount": "размер скидки строкой, например -23%",
  "id_sku": "артикул или SKU под штрих-кодом",
  "print_datetime": "дата и время печати как на ценнике, например 24.12.2025 12:25",
  "code": "код зоны полки если есть, например 01_025019 - 026015",
  "additional_info": "дополнительная информация: тип вина Сухое, номер весов и т.п.",
  "special_symbols": "буква Ш если есть символ шелфтокера",
  "wholesale_level_1_count": "минимальное количество для оптовой цены 1",
  "wholesale_level_1_price": "оптовая цена 1 — число с точкой",
  "wholesale_level_2_count": "минимальное количество для оптовой цены 2",
  "wholesale_level_2_price": "оптовая цена 2 — число с точкой"
}

Правила:
- Цены ТОЛЬКО числа с точкой: 72.59, а не 72,59 и не «72 рублей»
- Для отсутствующих полей используй null
- Не придумывай данные которых нет на ценнике
- Верни ТОЛЬКО JSON-объект, без пояснений, без markdown, без ```"""


# ════════════════════════════════════════════════════════════════════════════
#  ПРЕДОБРАБОТКА ИЗОБРАЖЕНИЯ
# ════════════════════════════════════════════════════════════════════════════

def upscale_if_small(img: np.ndarray, min_dim: int = 800) -> np.ndarray:
    """Увеличиваем маленькие изображения для лучшего OCR."""
    h, w = img.shape[:2]
    if min(h, w) < min_dim:
        scale = min_dim / min(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
    return img


def img_to_base64(img_bgr: np.ndarray) -> str:
    """Конвертируем numpy-изображение в base64 JPEG для Ollama."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ════════════════════════════════════════════════════════════════════════════
#  OLLAMA
# ════════════════════════════════════════════════════════════════════════════

def check_ollama(host: str, model: str):
    """Проверяем что Ollama запущена и модель доступна."""
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        r.raise_for_status()
        available = [m["name"] for m in r.json().get("models", [])]
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Ollama не запущена на {host}")
        print("   Запустите: ollama serve")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Ошибка подключения к Ollama: {e}")
        sys.exit(1)

    # Сначала точное совпадение, потом по базовому имени
    exact_match = model in available
    model_base = model.split(":")[0]
    partial_match = [m for m in available if m.startswith(model_base)]

    if not exact_match and not partial_match:
        print(f"\n❌ Модель '{model}' не найдена.")
        print(f"   Доступные: {available if available else 'нет моделей'}")
        print(f"   Скачайте:  ollama pull {model}")
        sys.exit(1)

    # Если точного совпадения нет — автоматически берём первую подходящую
    if not exact_match and partial_match:
        actual_model = partial_match[0]
        print(f"  ⚠️  Модель '{model}' не найдена точно, использую '{actual_model}'")
        # Подменяем модель глобально через возврат
        return actual_model

    print(f"  Ollama: {host}  |  Модель: {model}  ✓")


def extract_json_from_text(text: str) -> str:
    """
    Надёжно вытаскиваем JSON из ответа модели.
    Убираем markdown-обёртки, ищем первый полный {...}.
    """
    # Убираем markdown-блоки ```json ... ```
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    # Ищем JSON-объект — берём самый длинный найденный блок
    # (защита от случаев когда модель добавляет текст после JSON)
    best = None
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start:i + 1]
                if best is None or len(candidate) > len(best):
                    best = candidate
    return best or ""


def call_ollama(host: str, model: str, img_b64: str) -> dict:
    """
    Отправляем изображение в Ollama, получаем JSON с полями ценника.
    """
    payload = {
        "model": model,
        "prompt": PROMPT,
        "images": [img_b64],
        "stream": False,
        "options": {
            "temperature": 0.0,
            # ИСПРАВЛЕНО: было 600 — не хватало на полный JSON с русским текстом
            "num_predict": 1500,
            # ИСПРАВЛЕНО: убрали stop ["\n\n\n"] — он обрывал JSON раньше времени
        }
    }

    r = requests.post(
        f"{host}/api/generate",
        json=payload,
        timeout=300  # Qwen2.5-VL на CPU может быть медленнее llava
    )
    r.raise_for_status()

    raw = r.json().get("response", "").strip()

    json_str = extract_json_from_text(raw)
    if not json_str:
        raise ValueError(f"JSON не найден в ответе модели: {raw[:300]}")

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Пробуем починить обрезанный JSON — добавляем закрывающую скобку
        # (бывает если модель не уложилась в num_predict)
        fixed = json_str.rstrip().rstrip(",") + "\n}"
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            raise ValueError(f"Не удалось распарсить JSON: {e}\nОтвет: {json_str[:300]}")


# ════════════════════════════════════════════════════════════════════════════
#  QR И ШТРИХ-КОДЫ (pyzbar — работает без сети)
# ════════════════════════════════════════════════════════════════════════════

def decode_codes(img_bgr: np.ndarray) -> dict:
    result = {"qr_data": [], "barcodes": []}
    if not HAS_PYZBAR:
        return result

    for src in [img_bgr, cv2.bitwise_not(img_bgr)]:
        decoded = pyzbar_decode(src)
        if decoded:
            for obj in decoded:
                raw = obj.data.decode("utf-8", errors="ignore").strip()
                if obj.type == "QRCODE":
                    if raw not in result["qr_data"]:
                        result["qr_data"].append(raw)
                else:
                    if raw not in result["barcodes"]:
                        result["barcodes"].append(raw)
            break

    return result


def parse_qr_content(qr_strings: list) -> dict:
    """Парсим данные внутри QR-кода: штрих-код и цены."""
    out = {
        "qr_code_barcode": "нет",
        "price1_qr": "нет", "price2_qr": "нет",
        "price3_qr": "нет", "price4_qr": "нет",
        "action_price_qr": "нет", "action_code_qr": "нет",
    }
    if not qr_strings:
        return out

    raw = qr_strings[0]

    # Пробуем JSON
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            out["qr_code_barcode"] = str(data.get("barcode", data.get("ean", "нет")))
            for i, p in enumerate(data.get("prices", [])[:4], 1):
                out[f"price{i}_qr"] = str(p)
            return out
    except Exception:
        pass

    # Разделители | ; ,
    parts = re.split(r"[|;,]", raw)
    prices_found, barcode_found = [], None
    for p in parts:
        p = p.strip()
        if re.fullmatch(r"\d{8,18}", p):
            barcode_found = p
        elif re.fullmatch(r"\d{1,6}[.,]\d{2}", p):
            prices_found.append(p.replace(",", "."))

    if barcode_found:
        out["qr_code_barcode"] = barcode_found
    elif re.fullmatch(r"\d{8,18}", raw):
        out["qr_code_barcode"] = raw

    for i, pr in enumerate(prices_found[:4], 1):
        out[f"price{i}_qr"] = pr

    return out


# ════════════════════════════════════════════════════════════════════════════
#  ЦВЕТ РАМКИ
# ════════════════════════════════════════════════════════════════════════════

def detect_frame_color(img: np.ndarray) -> str:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    border = np.concatenate([
        hsv[:20, :].reshape(-1, 3), hsv[-20:, :].reshape(-1, 3),
        hsv[:, :20].reshape(-1, 3), hsv[:, -20:].reshape(-1, 3),
    ])
    h = float(np.median(border[:, 0]))
    s = float(np.median(border[:, 1]))
    v = float(np.median(border[:, 2]))

    if s < 40:
        return "white" if v > 128 else "нет"
    if h < 15 or h > 165: return "red"
    if 15 <= h < 35:       return "yellow"
    if 35 <= h < 85:       return "green"
    if 85 <= h < 130:      return "blue"
    return "нет"


# ════════════════════════════════════════════════════════════════════════════
#  СБОРКА СТРОКИ CSV
# ════════════════════════════════════════════════════════════════════════════

def build_row(filename: str, llm_data: dict, qr_fields: dict,
              barcode_scanner: str, color: str) -> dict:
    """Собираем финальную строку CSV из всех источников."""
    row = {col: "нет" for col in CSV_COLUMNS}
    row["filename"] = filename

    # Поля от нейросети
    llm_fields = {
        "product_name", "price_default", "price_card", "price_discount",
        "discount_amount", "id_sku", "print_datetime", "code",
        "additional_info", "special_symbols",
        "wholesale_level_1_count", "wholesale_level_1_price",
        "wholesale_level_2_count", "wholesale_level_2_price",
    }
    for key in llm_fields:
        val = llm_data.get(key)
        if val is not None and str(val).lower() not in ("null", "none", ""):
            row[key] = str(val)

    # Штрих-код: физический сканер приоритетнее OCR
    if barcode_scanner:
        row["barcode"] = barcode_scanner
    elif llm_data.get("barcode"):
        row["barcode"] = str(llm_data["barcode"])

    # QR-данные
    row.update(qr_fields)

    # Если штрих-код только в QR
    if row["barcode"] == "нет" and qr_fields["qr_code_barcode"] != "нет":
        row["barcode"] = qr_fields["qr_code_barcode"]

    row["color"] = color

    for k in ("frame_timestamp", "x_min", "y_min", "x_max", "y_max"):
        row[k] = "нет"

    return row


# ════════════════════════════════════════════════════════════════════════════
#  ОБРАБОТКА ОДНОГО ФАЙЛА
# ════════════════════════════════════════════════════════════════════════════

def process_image(img_path: Path, host: str, model: str) -> dict:
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError("Не удалось прочитать файл")

    img = upscale_if_small(img)

    # 1. QR и штрих-коды (pyzbar — быстро, без сети)
    codes = decode_codes(img)
    barcode_scanner = codes["barcodes"][0] if codes["barcodes"] else None
    qr_fields = parse_qr_content(codes["qr_data"])

    # 2. Нейросеть читает текст
    img_b64 = img_to_base64(img)
    llm_data = call_ollama(host, model, img_b64)

    # 3. Цвет рамки
    color = detect_frame_color(img)

    # 4. Собираем строку
    return build_row(img_path.name, llm_data, qr_fields, barcode_scanner, color)


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════

def collect_images(inputs: list) -> list:
    images = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.suffix.lower() in SUPPORTED_EXT:
                    images.append(f)
        elif p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
            images.append(p)
        else:
            print(f"  [!] Пропускаю: {inp}")
    seen = set()
    return [x for x in images if not (x in seen or seen.add(x))]


def main():
    parser = argparse.ArgumentParser(
        description="Парсер ценников: изображения → CSV (Ollama + Qwen2.5-VL)"
    )
    parser.add_argument("inputs", nargs="+",
                        help="Папка с изображениями или отдельные файлы")
    parser.add_argument("-o", "--output", default="output.csv",
                        help="Выходной CSV (по умолчанию: output.csv)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Модель Ollama (по умолчанию: {DEFAULT_MODEL})")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help=f"Адрес Ollama (по умолчанию: {DEFAULT_HOST})")
    args = parser.parse_args()

    images = collect_images(args.inputs)
    if not images:
        print("Ошибка: изображения не найдены.")
        sys.exit(1)

    print(f"\nНайдено изображений: {len(images)}")
    check_ollama(args.host, args.model)
    print(f"Выходной файл: {args.output}")
    print("⚠️  На CPU каждый ценник занимает ~60-180 сек (qwen2.5vl:7b)\n")

    rows, errors = [], []
    for i, img_path in enumerate(images, 1):
        print(f"  [{i}/{len(images)}] {img_path.name} ...", end=" ", flush=True)
        try:
            row = process_image(img_path, args.host, args.model)
            rows.append(row)
            print("✓")
        except json.JSONDecodeError as e:
            print(f"✗  (модель вернула не JSON: {e})")
            errors.append(img_path.name)
        except Exception as e:
            print(f"✗  ({e})")
            errors.append(img_path.name)

    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Записано строк: {len(rows)} → {args.output}")
    if errors:
        print(f"⚠️  Ошибки ({len(errors)}): {', '.join(errors)}")


if __name__ == "__main__":
    main()