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
   ollama pull qwen2.5vl:7b          # рекомендуется, ~5 GB
   ollama pull qwen2.5vl:3b-q4_K_M   # если мало RAM (~2.3 GB)

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
   python det.py ./папка --model qwen2.5vl:7b
   python det.py ./папка --host http://192.168.1.10:11434
   python det.py ./папка --retries 3        # повторы при ошибке парсинга JSON
   python det.py ./папка --no-few-shot      # отключить few-shot примеры
"""

import os
import re
import csv
import sys
import json
import base64
import argparse
import time
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
DEFAULT_MODEL   = "qwen2.5vl:7b"
DEFAULT_HOST    = "http://localhost:11434"
EXAMPLES_DIR    = Path(__file__).parent / "examples"
DEFAULT_RETRIES = 2   # повторов при неудачном JSON-парсинге

# ── Колонки CSV ───────────────────────────────────────────────────────────────
CSV_COLUMNS = [
    "filename", "tag_type",
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
#  УНИВЕРСАЛЬНОЕ ПРАВИЛО ЦЕН (вставляется в промпты где нужно)
# ════════════════════════════════════════════════════════════════════════════

PRICE_LOGIC_HINT = """
UNIVERSAL PRICE RULE (always apply this logic):
- price_default (Без карты) = HIGHER number = original price BEFORE discount
- price_card   (С картой)  = LOWER  number = discounted price WITH loyalty card
- VERIFY with math: price_default × (1 − discount%) ≈ price_card
  Example: 2421 × (1 − 0.32) = 1646 ≈ 1631 ✓
- If you see -46% circle and two numbers 5578 and 2999 → 5578×0.54=3012≈2999 ✓
  So price_default=5578, price_card=2999.99
- The BIG bold number filling most of the tag = price_card (С картой, already discounted)
- The SMALL number near top edge = price_default (Без карты, original higher price)
- NEVER swap them: price_card is ALWAYS smaller than price_default on discount tags
"""


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

PROMPT_VERIFY_WINE = (
    "Does this price tag contain a rounded rectangle box "
    "with a wine type word like Сухое, Полусладкое, Брют, Красное, Белое, Розовое? "
    "Reply YES or NO only."
)


# ════════════════════════════════════════════════════════════════════════════
#  ПРОМПТЫ ПО ТИПУ ЦЕННИКА
# ════════════════════════════════════════════════════════════════════════════

PROMPTS = {

# ── 1. Стандартный ценник ─────────────────────────────────────────────────
"standard": f"""This is a STANDARD Russian store price tag.
It has a discount circle (e.g. -32%), TWO prices, and a loyalty card system.
{PRICE_LOGIC_HINT}
Return ONLY a valid JSON object. No explanation, no markdown, no code blocks.

{{
  "product_name": "full product name with weight/volume and country in brackets",
  "price_default": "price WITHOUT card (Без карты) — HIGHER number — as decimal e.g. 250.09",
  "price_card":    "price WITH card (С картой) — LOWER number — as decimal e.g. 168.90",
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
}}

Key reading rules:
- Large digits + small superscript = one price: «168» + «⁹⁰» = 168.90
- Thousands separator space: «1 284» = 1284
- price_default is ALWAYS > price_card (higher original price vs lower discounted price)
- Use null for every field not visible on the tag
- Return ONLY the JSON object, nothing else""",


# ── 2. Весовой ценник ─────────────────────────────────────────────────────
"weight": f"""This is a WEIGHT/BULK Russian store price tag.
Prices are shown PER UNIT WEIGHT — per 100g ("за 100г") or per kg ("за кг" / "за 1 кг").
It may also have a discount circle on the left.
{PRICE_LOGIC_HINT}
Return ONLY a valid JSON object. No explanation, no markdown, no code blocks.

{{
  "product_name": "full name including grade/sort and weight category e.g. Креветки Королевские с/м с/г 50/70 вес (Россия)",
  "price_default": "price WITHOUT card PER UNIT WEIGHT — HIGHER number — e.g. 72.59",
  "price_card":    "price WITH card PER UNIT WEIGHT — LOWER number — e.g. 55.39",
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
}}

Key reading rules:
- Prices are per-unit-weight, NOT total product price
- Large digits + small superscript = one price: «55» + «³⁹» = 55.39
- Thousands separator space: «1 284» = 1284
- price_default is ALWAYS > price_card
- Use null for every field not visible on the tag
- Return ONLY the JSON object, nothing else""",


# ── 3. Оптовый ценник ────────────────────────────────────────────────────
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
"wine": f"""This is a WINE price tag from a Russian store.
It has a wine-type label in a rounded rectangle (e.g. "Сухое", "Полусладкое", "Брют"),
a discount circle on the left, and TWO prices.
{PRICE_LOGIC_HINT}
Return ONLY a valid JSON object. No explanation, no markdown, no code blocks.

{{
  "product_name": "full wine name including grape variety, style, colour, region and volume e.g. Вино HAUT MARIN Colombard Ugni-blanc Littorine ордин. бел. сух. (Франция) 0.75L",
  "price_default": "price WITHOUT card (Без карты) — HIGHER number near top-right — e.g. 2421.93",
  "price_card":    "price WITH card (С картой) — LOWER BIG bold dominant number — e.g. 1631.99",
  "price_discount": null,
  "discount_amount": "discount in circle e.g. -32%",
  "barcode":   "13-15 digit EAN barcode at the bottom",
  "id_sku":    "article/SKU small digits below or near discount circle",
  "print_datetime": "print date and time e.g. 28.04.2026 16:17",
  "code":      "shelf zone code if visible, else null",
  "additional_info": "WINE TYPE from the rounded rectangle box — required: e.g. Сухое or Полусладкое",
  "special_symbols": "Ш if shelf-talker circle present, else null",
  "wholesale_level_1_count": null,
  "wholesale_level_1_price": null,
  "wholesale_level_2_count": null,
  "wholesale_level_2_price": null
}}

Key reading rules:
- price_default (Без карты) is ALWAYS the HIGHER number — original price before discount
- price_card (С картой) is ALWAYS the LOWER number — price after discount
- The BIG bold number = price_card (С картой). The small top-right number = price_default (Без карты)
- Large digits + small superscript = one price: «1631» + «⁹⁹» = 1631.99
- Thousands separator space: «1 747» = 1747, «2 421» = 2421, «5 578» = 5578
- additional_info MUST contain the wine type from the rounded rectangle box
- Use null for every field not visible on the tag
- Return ONLY the JSON object, nothing else""",


# ── 5. Ценник с шелфтокером ───────────────────────────────────────────────
"shelftaker": f"""This is a SHELF-TAKER price tag from a Russian store.
It has TWO panels: left = main price tag, right = auxiliary panel with scale number
("номер на весах N") or promotional label ("Удачная упаковка").
{PRICE_LOGIC_HINT}
Return ONLY a valid JSON object. No explanation, no markdown, no code blocks.

{{
  "product_name": "full name from LEFT panel including grade/sort e.g. Орехи грецкие очищ. 1 сорт вес",
  "price_default": "price WITHOUT card (Без карты за кг) from LEFT panel — HIGHER — e.g. 1284.29",
  "price_card":    "price WITH card (С картой за кг) from LEFT panel — LOWER — e.g. 1029.99",
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
}}

Key reading rules:
- LEFT panel = product info and prices. RIGHT panel = auxiliary info → goes into additional_info
- Large digits + small superscript = one price: «1 029» + «⁹⁹» = 1029.99
- Thousands separator space: «1 284» = 1284, «1 029» = 1029
- price_default is ALWAYS > price_card
- Use null for every field not visible on the tag
- Return ONLY the JSON object, nothing else""",


# ── 6. Простой ценник ────────────────────────────────────────────────────
"simple": """This is a SIMPLE Russian store price tag.
No discount circle. Prices are listed plainly — "Без карты" and "По карте" (or just two price rows).

Return ONLY a valid JSON object. No explanation, no markdown, no code blocks.

{
  "product_name": "full product name with weight/volume and country e.g. Шоколад FAZER Geisha (Финляндия) 100г",
  "price_default": "price WITHOUT card (Без карты) — HIGHER — e.g. 345.09",
  "price_card":    "price WITH card (По карте) — LOWER — e.g. 303.79",
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
- price_default is ALWAYS >= price_card
- Use null for every field not visible on the tag
- Return ONLY the JSON object, nothing else""",

}


# ════════════════════════════════════════════════════════════════════════════
#  FEW-SHOT ЭТАЛОННЫЕ ОТВЕТЫ
# ════════════════════════════════════════════════════════════════════════════

FEW_SHOT_EXAMPLES = {
    "standard": {
        "desc": (
            "EXAMPLE — Standard price tag (Nescafe Classic coffee, -32%).\n"
            "KEY: small text near top 250.09 = price_default (Без карты, HIGHER original price).\n"
            "KEY: big bold 168.90 = price_card (С картой, LOWER discounted price).\n"
            "Verify: 250.09 × 0.68 = 170.06 ≈ 168.90 ✓"
        ),
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
        "desc": (
            "EXAMPLE — Weight price tag (Royal shrimps, price per 100g).\n"
            "KEY: price_default=72.59 (Без карты, HIGHER), price_card=55.39 (С картой, LOWER).\n"
            "Verify: 72.59 × 0.77 = 55.89 ≈ 55.39 ✓"
        ),
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
        "desc": (
            "EXAMPLE — Wine price tag (CITRAN Bordeaux, dry red, -27%).\n"
            "KEY: small top-right number 2631 = price_default (Без карты, HIGHER original price).\n"
            "KEY: big bold number 1899.99 = price_card (С картой, LOWER discounted price).\n"
            "Verify: 2631 × 0.73 = 1920.63 ≈ 1899.99 ✓\n"
            "NOTE: price_default is ALWAYS higher than price_card on wine tags."
        ),
        "json": {
            "product_name": "Вино CITRAN Бордо Супельор кр. сух. (Франция) 0.75L",
            "price_default": "2631.00", "price_card": "1899.99",
            "price_discount": None, "discount_amount": "-27%",
            "barcode": None, "id_sku": None,
            "print_datetime": None, "code": None,
            "additional_info": "Сухое", "special_symbols": None,
            "wholesale_level_1_count": None, "wholesale_level_1_price": None,
            "wholesale_level_2_count": None, "wholesale_level_2_price": None,
        }
    },
    "shelftaker": {
        "desc": (
            "EXAMPLE — Shelf-taker price tag (walnuts, right panel = scale number 214).\n"
            "KEY: price_default=1284.29 (HIGHER), price_card=1029.99 (LOWER)."
        ),
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

def upscale_if_small(img: np.ndarray, min_dim: int = 1000) -> np.ndarray:
    """Увеличиваем изображение если оно слишком маленькое.
    Порог поднят до 1000px — лучше читается мелкий текст."""
    h, w = img.shape[:2]
    if min(h, w) < min_dim:
        scale = min_dim / min(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
    return img


def sharpen(img: np.ndarray) -> np.ndarray:
    """Лёгкое повышение резкости — помогает читать мелкий текст."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)


def img_to_base64(img_bgr: np.ndarray, sharpen_img: bool = True) -> str:
    """Конвертируем изображение в base64 JPEG.
    sharpen=True — лёгкое повышение резкости перед отправкой модели."""
    if sharpen_img:
        img_bgr = sharpen(img_bgr)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=95)  # качество повышено с 92 до 95
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
                    img = upscale_if_small(img)
                    result[tag_type] = img_to_base64(img, sharpen_img=False)
                    break
    return result


# ════════════════════════════════════════════════════════════════════════════
#  ПРЕДОБРАБОТКА ДЛЯ ШТРИХ-КОДОВ
# ════════════════════════════════════════════════════════════════════════════

def preprocess_for_barcode(img: np.ndarray) -> list:
    """Возвращает несколько вариантов изображения для pyzbar.

    Ценники часто имеют неравномерное освещение и блики — поэтому
    пробуем несколько вариантов бинаризации и масштаба.
    """
    variants = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Выравнивание яркости через CLAHE (убирает блики)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    # 2. Простая бинаризация Otsu
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    # 3. Otsu после выравнивания яркости
    _, otsu_eq = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu_eq)

    # 4. Адаптивная бинаризация (помогает при неравномерном освещении)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    variants.append(adaptive)

    # 5. Адаптивная с большим блоком
    adaptive2 = cv2.adaptiveThreshold(
        equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 25, 10
    )
    variants.append(adaptive2)

    # 6. Увеличенный вариант × 2 (штрих-коды на фото бывают мелкими)
    h, w = img.shape[:2]
    big = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    _, big_otsu = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(big_otsu)

    # 7. Инверсия каждого базового варианта
    variants += [cv2.bitwise_not(v) for v in variants[:]]

    return variants


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


def call_ollama(host: str, model: str, prompt: str,
                images: list, timeout: int = 600) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "images": images,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 800,
            # Увеличиваем контекст — помогает при few-shot с двумя изображениями
            "num_ctx": 8192,
        },
    }
    r = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def classify_tag(host: str, model: str, img_b64: str) -> str:
    """Шаг 1: определяем тип ценника.

    После первичной классификации делаем дополнительную проверку:
    если результат standard, проверяем — не wine ли это на самом деле.
    Wine чаще всего путают со standard, это самая частая ошибка классификатора.
    """
    raw = call_ollama(host, model, PROMPT_CLASSIFY, [img_b64], timeout=600)
    tag_type = raw.strip().split()[0].lower() if raw.strip() else ""

    if tag_type not in PROMPTS:
        for t in PROMPTS:
            if t in raw.lower():
                tag_type = t
                break
        else:
            tag_type = "standard"  # fallback

    # Перепроверка: standard → wine?
    if tag_type == "standard":
        check = call_ollama(host, model, PROMPT_VERIFY_WINE, [img_b64], timeout=60)
        if "yes" in check.lower():
            tag_type = "wine"
            print(f"    → переклассифицирован в wine", end=" ", flush=True)

    return tag_type


def extract_fields(host: str, model: str, img_b64: str,
                   tag_type: str, use_few_shot: bool = True) -> dict:
    """Шаг 2: извлекаем поля специализированным промптом.

    При use_few_shot=True добавляем эталонное изображение (если есть).
    При ошибке парсинга — caller делает retry.
    """
    base_prompt = PROMPTS[tag_type]
    example_img = EXAMPLE_IMAGES.get(tag_type) if use_few_shot else None

    if example_img:
        ex = FEW_SHOT_EXAMPLES[tag_type]
        desc_text = ex["desc"]
        example_json = json.dumps(ex["json"], ensure_ascii=False, indent=2)
        prompt = (
            "You are given TWO images.\n\n"
            "IMAGE 1 is an EXAMPLE price tag. "
            f"Study it together with this correct JSON:\n"
            f"{desc_text}\n"
            f"{example_json}\n\n"
            "CRITICAL: Do NOT copy any values from IMAGE 1 into your answer. "
            "IMAGE 1 shows FORMAT, STRUCTURE, and price layout logic only.\n"
            "Pay special attention to the NOTE about which price is higher/lower.\n\n"
            "IMAGE 2 is the NEW price tag you must extract data from. "
            "Read IMAGE 2 carefully. Apply the same price-reading logic.\n\n"
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
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Попытка починить обрезанный JSON
        fixed = json_str.rstrip().rstrip(",") + "\n}"
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            raise ValueError(f"Не удалось распарсить JSON: {e}\n{json_str[:300]}")


# ════════════════════════════════════════════════════════════════════════════
#  ВЕРИФИКАЦИЯ ЦЕН
# ════════════════════════════════════════════════════════════════════════════

def _parse_price(val) -> float | None:
    """Парсим строку цены в float, None если не получилось."""
    if val is None:
        return None
    s = str(val).strip().replace(",", ".").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


def verify_prices(llm_data: dict, tag_type: str) -> dict:
    """Проверяем и при необходимости исправляем логику цен.

    Правило: price_default (Без карты) должна быть >= price_card (С картой).
    Если перепутаны — меняем местами.
    Дополнительно проверяем через процент скидки.
    """
    if tag_type not in ("standard", "wine", "weight", "simple", "shelftaker"):
        return llm_data

    data = llm_data.copy()
    pd_val = _parse_price(data.get("price_default"))
    pc_val = _parse_price(data.get("price_card"))

    if pd_val is None or pc_val is None or pd_val == 0 or pc_val == 0:
        return data

    # Если цены перепутаны местами — меняем
    if pd_val < pc_val:
        print(f"\n    ⚠️  Цены перепутаны ({pd_val} < {pc_val}), меняю местами", end="", flush=True)
        data["price_default"], data["price_card"] = data["price_card"], data["price_default"]
        pd_val, pc_val = pc_val, pd_val

    # Проверка через процент скидки
    discount_str = str(data.get("discount_amount") or "")
    m = re.search(r"(\d+)", discount_str)
    if m:
        discount_pct = int(m.group(1)) / 100
        expected_card = pd_val * (1 - discount_pct)
        deviation = abs(expected_card - pc_val) / pc_val if pc_val else 1

        if deviation > 0.20:
            # Отклонение > 20% — предупреждаем, но не меняем данные
            print(
                f"\n    ⚠️  Проверь вручную: ожидалась цена по карте ≈{expected_card:.2f}, "
                f"получено {pc_val} (откл. {deviation*100:.1f}%)",
                end="", flush=True,
            )

    return data


# ════════════════════════════════════════════════════════════════════════════
#  QR И ШТРИХ-КОДЫ
# ════════════════════════════════════════════════════════════════════════════

def decode_codes(img_bgr: np.ndarray) -> dict:
    """Читаем QR и штрих-коды через pyzbar.

    Перебираем несколько вариантов предобработки чтобы
    справиться с бликами и неравномерным освещением.
    """
    result = {"qr_data": [], "barcodes": []}
    if not HAS_PYZBAR:
        return result

    # Сначала пробуем оригинальное изображение
    all_variants = [img_bgr] + preprocess_for_barcode(img_bgr)

    for src in all_variants:
        try:
            decoded = pyzbar_decode(src)
        except Exception:
            continue
        for obj in decoded:
            raw = obj.data.decode("utf-8", errors="ignore").strip()
            if not raw:
                continue
            bucket = "qr_data" if obj.type == "QRCODE" else "barcodes"
            if raw not in result[bucket]:
                result[bucket].append(raw)

        # Если уже нашли и QR и штрих-код — не продолжаем
        if result["qr_data"] and result["barcodes"]:
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
    row["tag_type"] = tag_type

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

    # Приоритет: сканер > LLM
    if barcode_scanner:
        row["barcode"] = barcode_scanner
    elif llm_data.get("barcode") and str(llm_data["barcode"]).lower() not in ("null", "none", ""):
        row["barcode"] = str(llm_data["barcode"])

    row.update(qr_fields)
    if row["barcode"] == "нет" and qr_fields["qr_code_barcode"] != "нет":
        row["barcode"] = qr_fields["qr_code_barcode"]

    row["color"] = color
    for k in ("frame_timestamp", "x_min", "y_min", "x_max", "y_max"):
        row[k] = "нет"
    return row


# ════════════════════════════════════════════════════════════════════════════
#  ОБРАБОТКА ОДНОГО ФАЙЛА
# ════════════════════════════════════════════════════════════════════════════

def process_image(img_path: Path, host: str, model: str,
                  retries: int = DEFAULT_RETRIES,
                  use_few_shot: bool = True) -> dict:
    """Полная обработка одного ценника.

    retries — количество повторов при неудачном парсинге JSON.
    При повторе few-shot отключается (иногда лишний контекст мешает).
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError("Не удалось прочитать файл")
    img = upscale_if_small(img)

    codes = decode_codes(img)
    barcode_scanner = codes["barcodes"][0] if codes["barcodes"] else None
    qr_fields = parse_qr_content(codes["qr_data"])

    img_b64 = img_to_base64(img)

    # Шаг 1: классификация (с перепроверкой wine)
    tag_type = classify_tag(host, model, img_b64)

    # Шаг 2: извлечение полей с повторами при ошибке
    llm_data = None
    last_error = None
    for attempt in range(retries + 1):
        try:
            fs = use_few_shot and (attempt == 0)  # few-shot только на первой попытке
            llm_data = extract_fields(host, model, img_b64, tag_type, use_few_shot=fs)
            break
        except (ValueError, json.JSONDecodeError) as e:
            last_error = e
            if attempt < retries:
                print(f"\n    ↻ retry {attempt + 1}/{retries}: {str(e)[:60]}", end="", flush=True)
                time.sleep(1)

    if llm_data is None:
        raise ValueError(f"Все попытки ({retries + 1}) провалились: {last_error}")

    # Шаг 3: верификация и исправление цен
    llm_data = verify_prices(llm_data, tag_type)

    color = detect_frame_color(img)
    return build_row(img_path.name, tag_type, llm_data, qr_fields, barcode_scanner, color)


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
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                        help=f"Повторы при ошибке JSON (по умолчанию: {DEFAULT_RETRIES})")
    parser.add_argument("--no-few-shot", action="store_true",
                        help="Отключить few-shot примеры (быстрее, но менее точно)")
    args = parser.parse_args()

    images = collect_images(args.inputs)
    if not images:
        print("Ошибка: изображения не найдены.")
        sys.exit(1)

    print(f"\nНайдено изображений: {len(images)}")
    args.model = check_ollama(args.host, args.model)
    print(f"Выходной файл: {args.output}")
    print(f"Повторов при ошибке JSON: {args.retries}")

    global EXAMPLE_IMAGES
    if not args.no_few_shot:
        EXAMPLE_IMAGES = load_example_images()
        if EXAMPLE_IMAGES:
            print(f"  Few-shot эталоны: {list(EXAMPLE_IMAGES.keys())} ✓")
        else:
            print(f"  Few-shot: папка examples/ не найдена, работаем без примеров")
            print(f"  Совет: создайте examples/ с файлами standard.jpg, weight.jpg и т.д.")
    else:
        print(f"  Few-shot: отключён (--no-few-shot)")

    print(f"  Типы ценников: {list(PROMPTS.keys())}")
    print(f"  Каждый ценник = 2-3 запроса к модели\n")

    rows, errors = [], []
    for i, img_path in enumerate(images, 1):
        print(f"  [{i}/{len(images)}] {img_path.name}", end=" ", flush=True)
        t0 = time.time()
        try:
            row = process_image(
                img_path, args.host, args.model,
                retries=args.retries,
                use_few_shot=not args.no_few_shot,
            )
            rows.append(row)
            elapsed = time.time() - t0
            print(f"→ {row['tag_type']}  [{elapsed:.1f}s] ✓")
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