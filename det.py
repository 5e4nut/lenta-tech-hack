#!/usr/bin/env python3
"""
Парсер ценников → CSV с классификацией типа ценника.
Модель сначала определяет тип ценника, затем применяет
специализированный промпт для точного извлечения данных.

═══════════════════════════════════════════════════════
 УСТАНОВКА (один раз)
═══════════════════════════════════════════════════════

1. Ollama:
   Linux/macOS:  curl -fsSL https://ollama.ai/install.sh | sh
   Windows:      https://ollama.ai/download

2. Скачать модель:
   ollama pull qwen2.5vl:3b-q4_K_M   # ~2.3 GB — рекомендуется для 8 GB RAM

3. Python-зависимости:
   pip install opencv-python-headless pillow pyzbar numpy requests

4. pyzbar системная либа:
   Ubuntu/Debian:  sudo apt install libzbar0
   macOS:          brew install zbar
   Windows:        pip install pyzbar  (dll идёт в комплекте)

═══════════════════════════════════════════════════════
 СТРУКТУРА ПАПКИ С ЭТАЛОНАМИ (few-shot примеры)
═══════════════════════════════════════════════════════
Создайте папку examples/ рядом со скриптом и положите туда
по одному эталонному изображению каждого типа:

  examples/
    standard.jpg      — обычный ценник (цена без карты + по карте + скидка в кружке)
    weight.jpg        — весовой товар (цена за кг или за 100г)
    wholesale.jpg     — оптовые цены ("по карте от 5 шт")
    wine.jpg          — вино с типом ("Сухое", "Полусладкое" в рамке)
    shelftaker.jpg    — ценник с шелфтокером (правая половина с номером весов)
    simple.jpg        — простой ценник без скидки в кружке

Если папка/файлы не найдены — скрипт работает без few-shot примеров.

═══════════════════════════════════════════════════════
 ИСПОЛЬЗОВАНИЕ
═══════════════════════════════════════════════════════
   python det.py ./папка_с_ценниками
   python det.py ./папка -o result.csv
   python det.py img1.jpg img2.png -o result.csv
   python det.py ./папка --model qwen2.5vl:3b-q4_K_M
   python det.py ./папка --host http://192.168.1.10:11434
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

# ── Настройки ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "qwen2.5vl:7b"
DEFAULT_HOST  = "http://localhost:11434"
EXAMPLES_DIR  = Path(__file__).parent / "examples"

# ── Колонки CSV ───────────────────────────────────────────────────────────────
CSV_COLUMNS = [
    "filename",
    "product_name", "price_default", "price_card", "price_discount",
    "barcode", "discount_amount", "id_sku", "print_datetime", "code",
    "additional_info", "color", "special_symbols", "frame_timestamp",
    "x_min", "y_min", "x_max", "y_max",
    "qr_code_barcode", "price1_qr", "price2_qr", "price3_qr", "price4_qr",
    "wholesale_level_1_count", "wholesale_level_1_price",
    "wholesale_level_2_count", "wholesale_level_2_price",
    "action_price_qr", "action_code_qr",
]

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


# ════════════════════════════════════════════════════════════════════════════
#  КЛАССИФИКАТОР
# ════════════════════════════════════════════════════════════════════════════

PROMPT_CLASSIFY = """Look at this price tag image from a Russian grocery store.
Classify it into exactly ONE of these types:

  standard    — has a discount circle (e.g. -32%), shows "Без карты" and "С картой" prices
  weight      — price is per kg or per 100g ("за кг", "за 100г"), may also have discount circle
  wholesale   — has bulk/multi-unit pricing tiers ("по карте от N шт", 2-3 stacked prices)
  wine        — has a wine-type label in a rounded box (e.g. "Сухое", "Полусладкое", "Брют")
  shelftaker  — has a separate right panel with scale number ("номер на весах") or "Удачная упаковка"
  simple      — no discount circle, just one or two plain prices listed

Reply with ONLY the type name, nothing else. Example reply: standard"""


# ════════════════════════════════════════════════════════════════════════════
#  ПРОМПТЫ ПО ТИПУ ЦЕННИКА
# ════════════════════════════════════════════════════════════════════════════

PROMPTS = {

# ── 1. Стандартный ценник ─────────────────────────────────────────────────
# Пример: Кофе NESCAFE, Креветки (без веса)
# Признаки: кружок со скидкой (-32%), "Без карты" = price_default, "С картой" = price_card
"standard": """This is a STANDARD Russian store price tag.
It has a discount circle (e.g. -32%), a price WITHOUT loyalty card ("Без карты"),
and a price WITH loyalty card ("С картой" or "С карты").

Return ONLY a valid JSON object. No explanation, no markdown, no code blocks.

{
  "product_name": "full product name with weight/volume and country in brackets",
  "price_default": "price WITHOUT card as decimal e.g. 250.09",
  "price_card":    "price WITH card as decimal e.g. 168.90",
  "price_discount": null,
  "discount_amount": "discount in circle as string e.g. -32%",
  "barcode":   "13-15 digit EAN barcode at bottom of tag",
  "id_sku":    "article/SKU digits below the discount circle e.g. 110301 002876",
  "print_datetime": "print date and time as shown e.g. 24.12.2025 12:25",
  "code":      "shelf zone code if visible e.g. 06_062 003, else null",
  "additional_info": null,
  "special_symbols": "Ш if a shelf-talker circle with letter Ш is present, else null",
  "wholesale_level_1_count": null,
  "wholesale_level_1_price": null,
  "wholesale_level_2_count": null,
  "wholesale_level_2_price": null
}

Key reading rules:
- Large digits + small superscript together = one price: «168» + «⁹⁰» = 168.90
- «Без карты» label → price_default. «С картой» label → price_card
- Discount circle on the LEFT side → discount_amount as negative percent string
- Use null for every field not visible on the tag
- Return ONLY the JSON object, nothing else""",


# ── 2. Весовой ценник ─────────────────────────────────────────────────────
# Пример: Креветки Королевские за 100г, Орехи грецкие за кг
# Признаки: подпись "Без карты за 100г" / "С картой за 1 кг"
"weight": """This is a WEIGHT/BULK Russian store price tag.
Prices are shown PER UNIT WEIGHT — per 100g ("за 100г") or per kg ("за кг" / "за 1 кг").
It may also have a discount circle on the left.

Return ONLY a valid JSON object. No explanation, no markdown, no code blocks.

{
  "product_name": "full name including grade/sort and weight category e.g. Креветки Королевские с/м с/г 50/70 вес (Россия)",
  "price_default": "price WITHOUT card PER UNIT WEIGHT e.g. 72.59",
  "price_card":    "price WITH card PER UNIT WEIGHT e.g. 55.39",
  "price_discount": null,
  "discount_amount": "discount in circle e.g. -23%, or null if absent",
  "barcode":   "13-15 digit EAN barcode",
  "id_sku":    "article/SKU digits below discount circle",
  "print_datetime": "print date and time e.g. 24.12.2025 12:25",
  "code":      "shelf zone code if visible e.g. 06_062 003, else null",
  "additional_info": "unit weight label e.g. за 100г or за кг",
  "special_symbols": "Ш if shelf-talker circle present, else null",
  "wholesale_level_1_count": null,
  "wholesale_level_1_price": null,
  "wholesale_level_2_count": null,
  "wholesale_level_2_price": null
}

Key reading rules:
- Prices are per-unit-weight, NOT total product price
- Large digits + small superscript = one price: «55» + «³⁹» = 55.39
- Thousands separator space: «1 284» = 1284, so «1 284» + «²⁹» = 1284.29
- Use null for every field not visible on the tag
- Return ONLY the JSON object, nothing else""",


# ── 3. Оптовый ценник ────────────────────────────────────────────────────
# Пример: Черноголовка НеЛимонад — три строки цен: 527.39 / 500.99 / 168.76 от 5 шт
# Признаки: несколько цен друг под другом, надпись "По карте от N шт"
"wholesale": """This is a WHOLESALE/MULTI-TIER Russian store price tag.
It shows 2-3 stacked price rows with bulk discount tiers ("по карте от N шт").

Return ONLY a valid JSON object. No explanation, no markdown, no code blocks.

{
  "product_name": "full product name with volume and country e.g. Напиток безалкогольный ЧЕРНОГОЛОВКА НеЛимонад Оригинальный сильногаз. ПЭТ (Россия) 2L",
  "price_default": "highest price WITHOUT loyalty card (Без карты) e.g. 527.39",
  "price_card":    "mid price WITH card for single unit (По карте) e.g. 500.99",
  "price_discount": "lowest bulk price (По карте от N шт) e.g. 168.76",
  "discount_amount": "discount percent if shown in a circle e.g. -32%, else null",
  "barcode":   "13-15 digit EAN barcode if visible, else null",
  "id_sku":    "article/SKU digits if visible, else null",
  "print_datetime": "print date and time e.g. 24.01.2023 16:09",
  "code":      "shelf zone code if visible, else null",
  "additional_info": "bulk condition text e.g. от 5 шт",
  "special_symbols": "Ш if shelf-talker circle present, else null",
  "wholesale_level_1_count": "minimum quantity for bulk tier 1 as number e.g. 5",
  "wholesale_level_1_price": "bulk tier 1 price as decimal e.g. 168.76",
  "wholesale_level_2_count": "minimum quantity for tier 2 if present, else null",
  "wholesale_level_2_price": "tier 2 price if present, else null"
}

Key reading rules:
- Three price rows top→bottom: regular price / card price / bulk price
- Large digits + small superscript = one price: «168» + «⁷⁶» = 168.76
- wholesale_level_1_count comes from the "от N шт" label next to the lowest price
- Use null for every field not visible on the tag
- Return ONLY the JSON object, nothing else""",


# ── 4. Винный ценник ─────────────────────────────────────────────────────
# Пример: Вино HAUT MARIN — с прямоугольником "Сухое"
# Признаки: блок с типом вина ("Сухое" / "Полусладкое"), скидка в кружке
"wine": """This is a WINE price tag from a Russian store.
It has a wine-type label in a rounded rectangle (e.g. "Сухое", "Полусладкое", "Брют"),
and a discount circle on the left side.

Return ONLY a valid JSON object. No explanation, no markdown, no code blocks.

{
  "product_name": "full wine name including grape variety, style, colour, region and volume e.g. Вино HAUT MARIN Colombard Ugni-blanc Littorine ордин. бел. сух. (Франция) 0.75L",
  "price_default": "price WITHOUT card (Без карты) e.g. 1747.35",
  "price_card":    "price WITH card (С картой) e.g. 1104.99",
  "price_discount": null,
  "discount_amount": "discount in circle e.g. -36%",
  "barcode":   "13-15 digit EAN barcode",
  "id_sku":    "article/SKU digits below discount circle",
  "print_datetime": "print date and time e.g. 28.04.2026 16:17",
  "code":      "shelf zone code if visible e.g. 21_ФВН 032_1_2_2, else null",
  "additional_info": "WINE TYPE text from rounded rectangle box e.g. Сухое or Полусладкое",
  "special_symbols": "Ш if shelf-talker circle present, else null",
  "wholesale_level_1_count": null,
  "wholesale_level_1_price": null,
  "wholesale_level_2_count": null,
  "wholesale_level_2_price": null
}

Key reading rules:
- additional_info MUST contain the wine type from the rounded box — this is required
- Large digits + small superscript = one price: «1104» + «⁹⁹» = 1104.99
- Thousands separator space: «1 747» = 1747
- Use null for every field not visible on the tag
- Return ONLY the JSON object, nothing else""",


# ── 5. Ценник с шелфтокером ───────────────────────────────────────────────
# Пример: Орехи грецкие + правая панель "номер на весах 214"
# Признаки: правая половина = отдельная панель с доп. информацией
"shelftaker": """This is a SHELF-TAKER price tag from a Russian store.
It has TWO panels: left = main price tag, right = auxiliary panel with scale number
("номер на весах N") or promotional label ("Удачная упаковка").

Return ONLY a valid JSON object. No explanation, no markdown, no code blocks.

{
  "product_name": "full name from LEFT panel including grade/sort e.g. Орехи грецкие очищ. 1 сорт вес",
  "price_default": "price WITHOUT card (Без карты за кг) from LEFT panel e.g. 1284.29",
  "price_card":    "price WITH card (С картой за кг) from LEFT panel e.g. 1029.99",
  "price_discount": null,
  "discount_amount": "discount in circle if present, else null",
  "barcode":   "13-15 digit EAN barcode from LEFT panel",
  "id_sku":    "article/SKU digits from LEFT panel e.g. 430601 060367",
  "print_datetime": "print date and time e.g. 24.12.2025 12:46",
  "code":      "shelf zone code if visible, else null",
  "additional_info": "full content of RIGHT panel e.g. номер на весах 214, or Удачная упаковка",
  "special_symbols": "Ш if shelf-talker circle is present in the LEFT panel, else null",
  "wholesale_level_1_count": null,
  "wholesale_level_1_price": null,
  "wholesale_level_2_count": null,
  "wholesale_level_2_price": null
}

Key reading rules:
- LEFT panel = product info and prices. RIGHT panel = auxiliary info → goes into additional_info
- Large digits + small superscript = one price: «1 029» + «⁹⁹» = 1029.99
- Thousands separator space: «1 284» = 1284, «1 029» = 1029
- Use null for every field not visible on the tag
- Return ONLY the JSON object, nothing else""",


# ── 6. Простой ценник ────────────────────────────────────────────────────
# Пример: Шоколад FAZER — без кружка скидки, просто две цены
# Признаки: нет кружка скидки, цены указаны лаконично
"simple": """This is a SIMPLE Russian store price tag.
No discount circle. Prices are listed plainly — "Без карты" and "По карте" (or just two price rows).

Return ONLY a valid JSON object. No explanation, no markdown, no code blocks.

{
  "product_name": "full product name with weight/volume and country e.g. Шоколад FAZER Geisha (Финляндия) 100г",
  "price_default": "price WITHOUT card (Без карты) e.g. 345.09",
  "price_card":    "price WITH card (По карте) e.g. 303.79",
  "price_discount": null,
  "discount_amount": null,
  "barcode":   "13-15 digit EAN barcode",
  "id_sku":    "article/SKU digits e.g. 320203 000763",
  "print_datetime": "print date and time e.g. 24.12.2025 13:00",
  "code":      "shelf zone code if visible, else null",
  "additional_info": null,
  "special_symbols": null,
  "wholesale_level_1_count": null,
  "wholesale_level_1_price": null,
  "wholesale_level_2_count": null,
  "wholesale_level_2_price": null
}

Key reading rules:
- No discount circle → discount_amount is always null
- Large digits + small superscript = one price: «303» + «⁷⁹» = 303.79
- Use null for every field not visible on the tag
- Return ONLY the JSON object, nothing else""",

}


# ════════════════════════════════════════════════════════════════════════════
#  FEW-SHOT ЭТАЛОННЫЕ ОТВЕТЫ (используются когда есть examples/ изображения)
# ════════════════════════════════════════════════════════════════════════════

FEW_SHOT_EXAMPLES = {
    "standard": {
        "desc": "EXAMPLE — Standard price tag (Nescafe Classic coffee):",
        "json": {
            "product_name": "Кофе NESCAFE Classic (Россия) 500g",
            "price_default": "250.09", "price_card": "168.90",
            "price_discount": None, "discount_amount": "-32%",
            "barcode": "4606272000180", "id_sku": "110301 002876",
            "print_datetime": "24.12.2025 12:25", "code": None,
            "additional_info": None, "special_symbols": None,
            "wholesale_level_1_count": None, "wholesale_level_1_price": None,
            "wholesale_level_2_count": None, "wholesale_level_2_price": None,
        }
    },
    "weight": {
        "desc": "EXAMPLE — Weight price tag (Royal shrimps, price per 100g):",
        "json": {
            "product_name": "Креветки Королевские с/м с/г 50/70 вес (Россия)",
            "price_default": "72.59", "price_card": "55.39",
            "price_discount": None, "discount_amount": "-23%",
            "barcode": "2999990013252", "id_sku": "220301 664884",
            "print_datetime": "24.12.2025 12:25", "code": "06_062 003",
            "additional_info": "за 100г", "special_symbols": "Ш",
            "wholesale_level_1_count": None, "wholesale_level_1_price": None,
            "wholesale_level_2_count": None, "wholesale_level_2_price": None,
        }
    },
    "wholesale": {
        "desc": "EXAMPLE — Wholesale price tag (Chernogolovka drink, 3 price tiers):",
        "json": {
            "product_name": "Напиток безалкогольный ЧЕРНОГОЛОВКА НеЛимонад Оригинальный сильногаз. ПЭТ (Россия) 2L",
            "price_default": "527.39", "price_card": "500.99",
            "price_discount": "168.76", "discount_amount": None,
            "barcode": None, "id_sku": None,
            "print_datetime": "24.01.2023 16:09", "code": None,
            "additional_info": "от 5 шт", "special_symbols": None,
            "wholesale_level_1_count": "5", "wholesale_level_1_price": "168.76",
            "wholesale_level_2_count": None, "wholesale_level_2_price": None,
        }
    },
    "wine": {
        "desc": "EXAMPLE — Wine price tag (Haut Marin, dry white, -36%):",
        "json": {
            "product_name": "Вино HAUT MARIN Colombard Ugni-blanc Littorine ордин. бел. сух. (Франция) 0.75L",
            "price_default": "1747.35", "price_card": "1104.99",
            "price_discount": None, "discount_amount": "-36%",
            "barcode": "3760094282509", "id_sku": "270108 726573",
            "print_datetime": "28.04.2026 16:17", "code": "21_ФВН 032_1_2_2",
            "additional_info": "Сухое", "special_symbols": "Ш",
            "wholesale_level_1_count": None, "wholesale_level_1_price": None,
            "wholesale_level_2_count": None, "wholesale_level_2_price": None,
        }
    },
    "shelftaker": {
        "desc": "EXAMPLE — Shelf-taker price tag (walnuts, right panel = scale number 214):",
        "json": {
            "product_name": "Орехи грецкие очищ. 1 сорт вес",
            "price_default": "1284.29", "price_card": "1029.99",
            "price_discount": None, "discount_amount": None,
            "barcode": "2099999089583", "id_sku": "430601 060367",
            "print_datetime": "24.12.2025 12:46", "code": None,
            "additional_info": "номер на весах 214", "special_symbols": None,
            "wholesale_level_1_count": None, "wholesale_level_1_price": None,
            "wholesale_level_2_count": None, "wholesale_level_2_price": None,
        }
    },
    "simple": {
        "desc": "EXAMPLE — Simple price tag (Fazer Geisha chocolate, no discount circle):",
        "json": {
            "product_name": "Шоколад FAZER Geisha (Финляндия) 100г",
            "price_default": "345.09", "price_card": "303.79",
            "price_discount": None, "discount_amount": None,
            "barcode": "6411401015908", "id_sku": "320203 000763",
            "print_datetime": "24.12.2025 13:00", "code": None,
            "additional_info": None, "special_symbols": None,
            "wholesale_level_1_count": None, "wholesale_level_1_price": None,
            "wholesale_level_2_count": None, "wholesale_level_2_price": None,
        }
    },
}

# Загруженные few-shot изображения (base64). Заполняется в main().
EXAMPLE_IMAGES: dict = {}


# ════════════════════════════════════════════════════════════════════════════
#  ПРЕДОБРАБОТКА ИЗОБРАЖЕНИЯ
# ════════════════════════════════════════════════════════════════════════════

def upscale_if_small(img: np.ndarray, min_dim: int = 800) -> np.ndarray:
    h, w = img.shape[:2]
    if min(h, w) < min_dim:
        scale = min_dim / min(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
    return img


def img_to_base64(img_bgr: np.ndarray) -> str:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_example_images() -> dict:
    """Загружаем эталонные изображения из папки examples/."""
    result = {}
    if not EXAMPLES_DIR.exists():
        return result
    for tag_type in PROMPTS:
        for ext in SUPPORTED_EXT:
            p = EXAMPLES_DIR / f"{tag_type}{ext}"
            if p.exists():
                img = cv2.imread(str(p))
                if img is not None:
                    result[tag_type] = img_to_base64(upscale_if_small(img))
                    break
    return result


# ════════════════════════════════════════════════════════════════════════════
#  OLLAMA
# ════════════════════════════════════════════════════════════════════════════

def check_ollama(host: str, model: str) -> str:
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

    exact = model in available
    partial = [m for m in available if m.startswith(model.split(":")[0])]

    if not exact and not partial:
        print(f"\n❌ Модель '{model}' не найдена.")
        print(f"   Доступные: {available or 'нет моделей'}")
        print(f"   Скачайте:  ollama pull {model}")
        sys.exit(1)

    if not exact and partial:
        print(f"  ⚠️  '{model}' → использую '{partial[0]}'")
        return partial[0]

    print(f"  Ollama: {host}  |  Модель: {model}  ✓")
    return model


def extract_json(text: str) -> str:
    """Вытаскиваем JSON-объект из произвольного текста ответа модели."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text).strip()
    best, depth, start = None, 0, None
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


import time

# ── Счётчик запросов и сброс модели ──────────────────────────────────────────
_request_counter = 0
RESET_EVERY_N    = 10        # сбрасывать модель каждые N запросов
SLOW_THRESHOLD_S = 45        # если ответ дольше X сек — предупреждение


def _unload_model(host: str, model: str) -> None:
    """Выгружает модель из памяти Ollama (keep_alive=0), чтобы сбросить состояние."""
    try:
        requests.post(
            f"{host}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=30,
        )
        print(f"  [↺] Модель выгружена из памяти (сброс каждые {RESET_EVERY_N} запросов)")
    except Exception as e:
        print(f"  [!] Не удалось выгрузить модель: {e}")


def call_ollama(host: str, model: str, prompt: str,
                images: list, timeout: int = 600) -> str:
    global _request_counter

    # Сброс каждые RESET_EVERY_N запросов
    if _request_counter > 0 and _request_counter % RESET_EVERY_N == 0:
        _unload_model(host, model)
        time.sleep(2)  # дать время на выгрузку

    _request_counter += 1

    payload = {
        "model": model,
        "prompt": prompt,
        "images": images,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 700},
    }

    t0 = time.time()
    r = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
    elapsed = time.time() - t0

    if elapsed > SLOW_THRESHOLD_S:
        print(f"  [⚠] Медленный ответ: {elapsed:.1f}s (порог {SLOW_THRESHOLD_S}s) — возможна деградация")

    r.raise_for_status()
    return r.json().get("response", "").strip()


def classify_tag(host: str, model: str, img_b64: str) -> str:
    """Шаг 1: определяем тип ценника одним коротким запросом."""
    raw = call_ollama(host, model, PROMPT_CLASSIFY, [img_b64], timeout=180)
    tag_type = raw.strip().split()[0].lower() if raw.strip() else ""
    if tag_type in PROMPTS:
        return tag_type
    for t in PROMPTS:
        if t in raw.lower():
            return t
    return "standard"  # fallback на самый частый тип


NULLED_FIELDS = {
    "product_name", "price_default", "price_card", "price_discount",
    "discount_amount", "barcode", "id_sku", "print_datetime", "code",
    "additional_info", "special_symbols",
    "wholesale_level_1_count", "wholesale_level_1_price",
    "wholesale_level_2_count", "wholesale_level_2_price",
}

def _is_full_copy(result: dict, example: dict) -> bool:
    """Возвращает True если ВСЕ ненулевые поля результата совпадают с примером."""
    compared = 0
    matched = 0
    for key in NULLED_FIELDS:
        ex_val = example.get(key)
        res_val = result.get(key)
        if ex_val is None:
            continue
        compared += 1
        if str(res_val).strip() == str(ex_val).strip():
            matched += 1
    if compared == 0:
        return False
    return matched == compared


def extract_fields(host: str, model: str, img_b64: str, tag_type: str) -> dict:
    """Шаг 2: извлекаем поля специализированным промптом."""
    base_prompt = PROMPTS[tag_type]
    example_img = EXAMPLE_IMAGES.get(tag_type)

    if example_img:
        ex = FEW_SHOT_EXAMPLES[tag_type]
        example_json = json.dumps(ex["json"], ensure_ascii=False, indent=2)
        prompt = (
            f"{ex['desc']}\n{example_json}\n\n"
            f"⚠️ WARNING: The JSON above shows the OUTPUT FORMAT only. "
            f"All values in it are FICTIONAL and belong to a DIFFERENT price tag.\n"
            f"Do NOT copy any value from the example into your answer.\n"
            f"Extract ONLY what you can literally read from the NEW image.\n"
            f"If you cannot read a field → set it to null.\n\n"
            f"Now extract data from the NEW price tag below.\n\n"
            f"{base_prompt}"
        )
        images = [example_img, img_b64]
    else:
        prompt = base_prompt
        images = [img_b64]

    raw = call_ollama(host, model, prompt, images, timeout=300)
    json_str = extract_json(raw)
    if not json_str:
        raise ValueError(f"JSON не найден в ответе: {raw[:300]}")

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        fixed = json_str.rstrip().rstrip(",") + "\n}"
        try:
            parsed = json.loads(fixed)
        except json.JSONDecodeError:
            raise ValueError(f"Не удалось распарсить JSON: {e}\n{json_str[:300]}")

    # Если модель скопировала все значения из примера — обнуляем всё
    if example_img and _is_full_copy(parsed, FEW_SHOT_EXAMPLES[tag_type]["json"]):
        print("  [!] Обнаружено полное копирование примера → все поля = null")
        for key in NULLED_FIELDS:
            parsed[key] = None

    return parsed


# ════════════════════════════════════════════════════════════════════════════
#  QR И ШТРИХ-КОДЫ
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
                bucket = "qr_data" if obj.type == "QRCODE" else "barcodes"
                if raw not in result[bucket]:
                    result[bucket].append(raw)
            break
    return result


def parse_qr_content(qr_strings: list) -> dict:
    out = {
        "qr_code_barcode": "нет",
        "price1_qr": "нет", "price2_qr": "нет",
        "price3_qr": "нет", "price4_qr": "нет",
        "action_price_qr": "нет", "action_code_qr": "нет",
    }
    if not qr_strings:
        return out
    raw = qr_strings[0]
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            out["qr_code_barcode"] = str(data.get("barcode", data.get("ean", "нет")))
            for i, p in enumerate(data.get("prices", [])[:4], 1):
                out[f"price{i}_qr"] = str(p)
            return out
    except Exception:
        pass
    parts = re.split(r"[|;,]", raw)
    prices, barcode = [], None
    for p in parts:
        p = p.strip()
        if re.fullmatch(r"\d{8,18}", p):
            barcode = p
        elif re.fullmatch(r"\d{1,6}[.,]\d{2}", p):
            prices.append(p.replace(",", "."))
    if barcode:
        out["qr_code_barcode"] = barcode
    elif re.fullmatch(r"\d{8,18}", raw):
        out["qr_code_barcode"] = raw
    for i, pr in enumerate(prices[:4], 1):
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
    if s < 40:      return "white" if v > 128 else "нет"
    if h < 15 or h > 165: return "red"
    if 15 <= h < 35:      return "yellow"
    if 35 <= h < 85:      return "green"
    if 85 <= h < 130:     return "blue"
    return "нет"


# ════════════════════════════════════════════════════════════════════════════
#  СБОРКА СТРОКИ CSV
# ════════════════════════════════════════════════════════════════════════════

def build_row(filename: str, tag_type: str, llm_data: dict,
              qr_fields: dict, barcode_scanner: str, color: str) -> dict:
    row = {col: "нет" for col in CSV_COLUMNS}
    row["filename"] = filename
    # tag_type не входит в CSV_COLUMNS (убран из эталона), но сохраняем
    # во внутреннем ключе чтобы process_image мог его логировать
    row["_tag_type"] = tag_type

    for key in {
        "product_name", "price_default", "price_card", "price_discount",
        "discount_amount", "id_sku", "print_datetime", "code",
        "additional_info", "special_symbols",
        "wholesale_level_1_count", "wholesale_level_1_price",
        "wholesale_level_2_count", "wholesale_level_2_price",
    }:
        val = llm_data.get(key)
        if val is not None and str(val).lower() not in ("null", "none", ""):
            row[key] = str(val)

    if barcode_scanner:
        row["barcode"] = barcode_scanner
    elif llm_data.get("barcode") and str(llm_data["barcode"]).lower() not in ("null", "none", ""):
        row["barcode"] = str(llm_data["barcode"])

    row.update(qr_fields)
    if row["barcode"] == "нет" and qr_fields["qr_code_barcode"] != "нет":
        row["barcode"] = qr_fields["qr_code_barcode"]

    row["color"] = color
    # frame_timestamp, x_min..y_max заполняются снаружи (в process_numpy_frame)
    return row


# ════════════════════════════════════════════════════════════════════════════
#  ОБРАБОТКА ОДНОГО ФАЙЛА
# ════════════════════════════════════════════════════════════════════════════

def process_image(img_path: Path, host: str, model: str) -> dict:
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError("Не удалось прочитать файл")
    img = upscale_if_small(img)

    codes = decode_codes(img)
    barcode_scanner = codes["barcodes"][0] if codes["barcodes"] else None
    qr_fields = parse_qr_content(codes["qr_data"])

    img_b64 = img_to_base64(img)

    # Шаг 1: классификация
    tag_type = classify_tag(host, model, img_b64)

    # Шаг 2: извлечение полей
    llm_data = extract_fields(host, model, img_b64, tag_type)

    color = detect_frame_color(img)
    return build_row(img_path.name, tag_type, llm_data, qr_fields, barcode_scanner, color)



# ════════════════════════════════════════════════════════════════════════════
#  ПУБЛИЧНЫЙ API ДЛЯ ВЫЗОВА ИЗ VIDEO-СКРИПТА
# ════════════════════════════════════════════════════════════════════════════

def process_numpy_frame(
    img: np.ndarray,
    frame_filename: str,
    host: str = DEFAULT_HOST,
    model: str = DEFAULT_MODEL,
    frame_timestamp: float = 0.0,
    bbox: tuple = None,
    video_name: str = None,
    frame_index: int = None,
) -> dict:
    """
    Обработать crop ценника, переданный как numpy BGR array (из video_detect.py).

    Параметры
    ----------
    img              : BGR numpy array (уже вырезанный crop ценника)
    frame_filename   : используется только для логирования (например "frame_00123_track7")
    host             : адрес Ollama
    model            : модель Ollama
    frame_timestamp  : время кадра в секундах (для обратной совместимости, не пишется в CSV)
    bbox             : (x_min, y_min, x_max, y_max) координаты float в оригинальном кадре
    video_name       : имя видеофайла — пишется в колонку filename (например "26_12-20.mp4")
    frame_index      : номер кадра — пишется в колонку frame_timestamp
    """
    img = upscale_if_small(img)

    codes = decode_codes(img)
    barcode_scanner = codes["barcodes"][0] if codes["barcodes"] else None
    qr_fields = parse_qr_content(codes["qr_data"])

    img_b64 = img_to_base64(img)
    tag_type = classify_tag(host, model, img_b64)
    llm_data = extract_fields(host, model, img_b64, tag_type)

    color = detect_frame_color(img)

    # filename в CSV = имя видеофайла, а не имя кадра
    csv_filename = video_name if video_name else frame_filename
    row = build_row(csv_filename, tag_type, llm_data, qr_fields, barcode_scanner, color)

    # frame_timestamp в эталоне = номер кадра (целое число)
    row["frame_timestamp"] = str(frame_index) if frame_index is not None else str(int(round(frame_timestamp * 30)))

    # bbox пишем как float с 1 знаком после запятой (формат эталона: 2011.9)
    if bbox:
        row["x_min"] = f"{float(bbox[0]):.1f}"
        row["y_min"] = f"{float(bbox[1]):.1f}"
        row["x_max"] = f"{float(bbox[2]):.1f}"
        row["y_max"] = f"{float(bbox[3]):.1f}"

    return row


def init_det_module(host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL) -> str:
    """
    Инициализация: загрузить few-shot примеры и проверить Ollama.
    Вызывать один раз при старте video_detect.py.
    Возвращает финальное имя модели (после auto-resolve).
    """
    global EXAMPLE_IMAGES
    EXAMPLE_IMAGES = load_example_images()
    return check_ollama(host, model)


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
        description="Парсер ценников с классификацией типа (Ollama + Qwen2.5-VL)"
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
    args.model = check_ollama(args.host, args.model)
    print(f"Выходной файл: {args.output}")

    global EXAMPLE_IMAGES
    EXAMPLE_IMAGES = load_example_images()
    if EXAMPLE_IMAGES:
        print(f"  Few-shot эталоны: {list(EXAMPLE_IMAGES.keys())} ✓")
    else:
        print(f"  Few-shot: папка examples/ не найдена, работаем без примеров")
        print(f"  Совет: создайте examples/ с файлами standard.jpg, weight.jpg и т.д.")

    print(f"  Типы ценников: {list(PROMPTS.keys())}")
    print(f"  Каждый ценник = 2 запроса к модели (~1-3 мин на CPU)\n")

    rows, errors = [], []
    for i, img_path in enumerate(images, 1):
        print(f"  [{i}/{len(images)}] {img_path.name}", end=" ", flush=True)
        try:
            row = process_image(img_path, args.host, args.model)
            rows.append(row)
            print(f"→ {row['tag_type']} ✓")
        except json.JSONDecodeError as e:
            print(f"✗  (не JSON: {e})")
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