"""
analyzer.py — Модуль аналізу зображень для AutoPhotoSorter.

Виконує:
  - Визначення білого/однотонного фону (OpenCV)
  - Виявлення тексту і водяних знаків (pytesseract OCR, опціонально)
  - Класифікацію контенту (Google Gemini API / OpenAI Vision API / локальна CLIP модель)

Категорії зображень (у порядку пріоритету для сортування):
  main        — Головне фото: білий/світлий фон (не обов'язково ідеально чистий),
                може бути присутній логотип або брендування (до 6 слів тексту),
                але це має бути саме фото товару
  packshot    — Пекшот: однотонний фон, інший ракурс
  detail      — Деталь: макрозйомка матеріалу або елементу
  lifestyle   — Лайфстайл: товар в інтер'єрі або в умовах реального використання
  kit         — Комплектація: товар з коробкою або аксесуарами
  infographic — Інфографіка: фото з розмірами, схемами, текстом характеристик
"""

import json
import os
import base64
import logging
import re
import threading
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Опціональні залежності — graceful degradation
# ---------------------------------------------------------------------------
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    logger.info("pytesseract не встановлено — OCR вимкнено.")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.info("google-generativeai не встановлено — Gemini API недоступний.")

try:
    from openai import OpenAI as _OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.info("openai не встановлено — OpenAI API недоступний.")

try:
    import torch
    import clip as clip_module
    CLIP_AVAILABLE = True
except (ImportError, Exception):
    CLIP_AVAILABLE = False
    logger.info("torch/CLIP не встановлено — локальна CLIP модель недоступна.")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.info("requests не встановлено — Ollama API недоступний.")

OLLAMA_URL_ENV_VAR = "AUTOPHOTOSORTER_OLLAMA_URL"
OLLAMA_MODEL_ENV_VAR = "AUTOPHOTOSORTER_OLLAMA_MODEL"
_DEFAULT_OLLAMA_URL_FALLBACK = "http://localhost:11434"
_DEFAULT_OLLAMA_MODEL_FALLBACK = "llava"
DEFAULT_OLLAMA_URL = os.environ.get(OLLAMA_URL_ENV_VAR, _DEFAULT_OLLAMA_URL_FALLBACK).strip() or _DEFAULT_OLLAMA_URL_FALLBACK
DEFAULT_OLLAMA_MODEL = os.environ.get(OLLAMA_MODEL_ENV_VAR, _DEFAULT_OLLAMA_MODEL_FALLBACK).strip() or _DEFAULT_OLLAMA_MODEL_FALLBACK
_OLLAMA_LOGGED_ISSUES: set[tuple[str, ...]] = set()
_OLLAMA_LOGGED_ISSUES_LOCK = threading.Lock()

# Ollama API endpoints
OLLAMA_TAGS_ENDPOINT = "/api/tags"
OLLAMA_GENERATE_ENDPOINT = "/api/generate"

# Максимальна кількість зображень в одному запиті ранжування до Ollama.
# Більше зображень — більший payload і час обробки.
OLLAMA_MAX_IMAGES_PER_RANK = 20

# Промти для AI (можуть бути змінені через set_custom_prompts)
_current_classify_prompt = None
_current_rank_prompt = None
_prompt_lock = threading.Lock()

def set_custom_prompts(classify_prompt: str | None = None, rank_prompt: str | None = None) -> None:
    """
    Встановлює користувацькі промти для класифікації та ранжування.
    
    Параметри:
      classify_prompt — промт для класифікації зображень (None = використати стандартний)
      rank_prompt — промт для ранжування зображень (None = використати стандартний)
    """
    global _current_classify_prompt, _current_rank_prompt
    with _prompt_lock:
        _current_classify_prompt = classify_prompt
        _current_rank_prompt = rank_prompt

def _get_classify_prompt() -> str:
    """Повертає активний промт для класифікації."""
    with _prompt_lock:
        return _current_classify_prompt if _current_classify_prompt else _CLASSIFY_PROMPT

def _get_rank_prompt_template() -> str:
    """Повертає активний шаблон промту для ранжування."""
    with _prompt_lock:
        return _current_rank_prompt if _current_rank_prompt else _RANK_PROMPT_TEMPLATE

# ---------------------------------------------------------------------------
# Константи категорій (порядок — пріоритет сортування)
# ---------------------------------------------------------------------------
CATEGORY_MAIN = "main"
CATEGORY_PACKSHOT = "packshot"
CATEGORY_DETAIL = "detail"
CATEGORY_LIFESTYLE = "lifestyle"
CATEGORY_KIT = "kit"
CATEGORY_INFOGRAPHIC = "infographic"

CATEGORY_ORDER = [
    CATEGORY_MAIN,
    CATEGORY_PACKSHOT,
    CATEGORY_DETAIL,
    CATEGORY_LIFESTYLE,
    CATEGORY_KIT,
    CATEGORY_INFOGRAPHIC,
]

CATEGORY_LABELS_UK = {
    CATEGORY_MAIN: "Головне фото (білий/світлий фон)",
    CATEGORY_PACKSHOT: "Пекшот",
    CATEGORY_DETAIL: "Деталь",
    CATEGORY_LIFESTYLE: "Лайфстайл",
    CATEGORY_KIT: "Комплектація",
    CATEGORY_INFOGRAPHIC: "Інфографіка",
}

# Підтримувані розширення зображень
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

# Налаштування класифікації
# Максимальна кількість слів тексту, що допускається для категорії 'main'
# Фото з більшою кількістю тексту будуть класифіковані як 'packshot' або 'infographic'
MAX_WORDS_FOR_MAIN_CATEGORY = 6

# ---------------------------------------------------------------------------
# 1. Визначення білого фону (OpenCV)
# ---------------------------------------------------------------------------

def detect_white_background(image_path: str,
                             pixel_threshold: int = 240,
                             border_fraction: float = 0.10) -> float:
    """
    Обчислює оцінку "білизни" фону зображення (від 0.0 до 1.0).

    Алгоритм:
      1. Читає зображення у відтінках сірого.
      2. Аналізує рамку шириною `border_fraction` від кожного краю.
      3. Рахує частку пікселів, яскравіших за `pixel_threshold`.
      4. Комбінує оцінку рамки (70%) і загальну яскравість (30%).

    Параметри налаштування:
      pixel_threshold  — поріг яскравості для "білого" пікселя (0–255).
                         За замовчуванням 240. Зменшіть до ~220 для
                         прийняття кремово-білих фонів.
      border_fraction  — частка розміру зображення для аналізу рамки (0–0.5).
                         За замовчуванням 0.10 (10%).
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return 0.0

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        bh = max(1, int(h * border_fraction))
        bw = max(1, int(w * border_fraction))

        border_pixels = np.concatenate([
            gray[:bh, :].flatten(),
            gray[-bh:, :].flatten(),
            gray[:, :bw].flatten(),
            gray[:, -bw:].flatten(),
        ])

        border_white = np.sum(border_pixels >= pixel_threshold) / len(border_pixels)
        overall_white = np.sum(gray >= pixel_threshold) / (h * w)

        return float(border_white * 0.7 + overall_white * 0.3)

    except Exception as exc:
        logger.warning("detect_white_background: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# 2. Виявлення тексту / водяних знаків (OCR)
# ---------------------------------------------------------------------------

def detect_text_or_watermarks(image_path: str) -> tuple[bool, str]:
    """
    Визначає наявність тексту або водяних знаків за допомогою OCR.

    Повертає (has_text: bool, detected_text: str).
    Якщо pytesseract недоступний — завжди повертає (False, '').

    Tesseract встановлюється окремо:
      Windows: https://github.com/UB-Mannheim/tesseract/wiki
      Linux:   sudo apt-get install tesseract-ocr
      macOS:   brew install tesseract
    """
    if not PYTESSERACT_AVAILABLE:
        return False, ""

    try:
        img = Image.open(image_path)
        # lang: 'eng' — англійська; можна додати '+ukr' якщо встановлено ukr пакет
        raw = pytesseract.image_to_string(img, lang="eng")
        words = [w for w in raw.split() if len(w) > 2]
        cleaned = " ".join(words)
        has_text = len(cleaned) > 5
        return has_text, cleaned

    except Exception as exc:
        logger.warning("detect_text_or_watermarks: %s", exc)
        return False, ""


# ---------------------------------------------------------------------------
# 3. Кодування зображення у base64 для API
# ---------------------------------------------------------------------------

def _encode_image_base64(image_path: str,
                          max_size: tuple = (1024, 1024)) -> str | None:
    """Кодує зображення у base64 JPEG для передачі в API."""
    try:
        img = Image.open(image_path)
        img.thumbnail(max_size, Image.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as exc:
        logger.warning("_encode_image_base64: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 4. Класифікація через Google Gemini API
# ---------------------------------------------------------------------------

_CLASSIFY_PROMPT = """Classify this product photo into exactly one of these categories:
1. main        — product on a white or very light background (doesn't need to be perfectly clean, small logos or branding in the background are acceptable - up to 6 words of text), the product itself should be clearly visible and the main focus, well-lit professional product photography
2. packshot    — product on a solid/neutral (not necessarily white) background, showing a different angle or composition
3. detail      — close-up or macro shot of product details, textures, or materials
4. lifestyle   — product placed in an interior setting or shown in real-life usage context
5. kit         — product shown together with its box, packaging, or all included accessories
6. infographic — image overlaid with text, dimensions, technical specs, callouts, or diagrams

Respond with ONLY one word — the category name (main / packshot / detail / lifestyle / kit / infographic)."""

_VALID_CATEGORIES = set(CATEGORY_ORDER)


def _parse_ai_response(text: str) -> str | None:
    """Парсить відповідь AI і повертає валідну категорію або None."""
    candidate = text.strip().lower()
    if candidate in _VALID_CATEGORIES:
        return candidate
    for cat in _VALID_CATEGORIES:
        if cat in candidate:
            return cat
    return None


def classify_with_gemini(image_path: str, api_key: str) -> str | None:
    """
    Класифікує зображення через Google Gemini API.

    Налаштування:
      api_key — ваш Gemini API ключ (https://aistudio.google.com/app/apikey).
      Модель: gemini-1.5-flash (швидка і дешева). Змініть на 'gemini-1.5-pro'
              для кращої точності.

    Повертає рядок категорії або None при помилці.
    """
    if not GEMINI_AVAILABLE:
        logger.error("google-generativeai не встановлено. Виконайте: pip install google-generativeai")
        return None
    try:
        genai.configure(api_key=api_key)
        # ---- Модель: змініть тут за потреби ----
        model = genai.GenerativeModel("gemini-1.5-flash")
        img = Image.open(image_path)
        response = model.generate_content([_get_classify_prompt(), img])
        return _parse_ai_response(response.text)
    except Exception as exc:
        logger.warning("classify_with_gemini: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 5. Класифікація через OpenAI Vision API
# ---------------------------------------------------------------------------

def classify_with_openai(image_path: str, api_key: str) -> str | None:
    """
    Класифікує зображення через OpenAI GPT-4o Vision API.

    Налаштування:
      api_key — ваш OpenAI API ключ (https://platform.openai.com/api-keys).
      Модель: gpt-4o-mini (баланс між ціною та якістю). Змініть на 'gpt-4o'
              для кращої точності.

    Повертає рядок категорії або None при помилці.
    """
    if not OPENAI_AVAILABLE:
        logger.error("openai не встановлено. Виконайте: pip install openai")
        return None
    try:
        client = _OpenAI(api_key=api_key)
        img_b64 = _encode_image_base64(image_path)
        if not img_b64:
            return None

        # ---- Модель: змініть тут за потреби ----
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _get_classify_prompt()},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ],
            }],
            max_tokens=50,
        )
        return _parse_ai_response(response.choices[0].message.content)
    except Exception as exc:
        logger.warning("classify_with_openai: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 6. Класифікація через Ollama (локальні моделі, наприклад Gemma)
# ---------------------------------------------------------------------------

def _normalize_ollama_base_url(ollama_url: str) -> str:
    normalized = ollama_url.strip().rstrip("/")
    for suffix in ("/api/generate", "/api/chat", "/api"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def _get_ollama_error_details(response) -> str:
    if response is None:
        return ""
    try:
        payload = response.json()
    except ValueError:
        return (response.text or "").strip()

    if isinstance(payload, dict):
        for key in ("error", "message"):
            value = payload.get(key)
            if value:
                return str(value).strip()
    return str(payload).strip()


def _log_ollama_issue_once(issue_key: tuple[str, ...], message: str, *args) -> None:
    with _OLLAMA_LOGGED_ISSUES_LOCK:
        if issue_key in _OLLAMA_LOGGED_ISSUES:
            return
        _OLLAMA_LOGGED_ISSUES.add(issue_key)
    logger.warning(message, *args)


def get_ollama_models(ollama_url: str | None) -> list[str]:
    """
    Отримує список доступних моделей з Ollama.
    
    Параметри:
      ollama_url — URL Ollama API (наприклад, http://localhost:11434)
    
    Повертає список назв моделей або порожній список при помилці.
    """
    if not REQUESTS_AVAILABLE:
        logger.error("requests не встановлено. Виконайте: pip install requests")
        return []
    
    if not ollama_url:
        logger.error("Ollama URL не вказано")
        return []
    
    ollama_url = _normalize_ollama_base_url(ollama_url)
    
    try:
        endpoint = f"{ollama_url}{OLLAMA_TAGS_ENDPOINT}"
        response = requests.get(endpoint, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        models = result.get("models", [])
        
        # Повертаємо список назв моделей
        model_names = []
        for model in models:
            if isinstance(model, dict) and "name" in model:
                model_names.append(model["name"])
        
        return sorted(model_names)
    except Exception as exc:
        logger.warning("get_ollama_models: Помилка отримання списку моделей з %s: %s", ollama_url, exc)
        return []


def classify_with_ollama(image_path: str, ollama_url: str | None, model_name: str = DEFAULT_OLLAMA_MODEL) -> str | None:
    """
    Класифікує зображення через Ollama (локальні моделі).

    Налаштування:
      ollama_url — URL Ollama API (наприклад, http://localhost:11434)
      model_name — назва моделі в Ollama з підтримкою зображень
                   (наприклад, "llava", "bakllava")

    Примітка: Модель повинна підтримувати обробку зображень.
    Моделі тільки для тексту (як чиста Gemma) не підійдуть.

    Повертає рядок категорії або None при помилці.
    """
    if not REQUESTS_AVAILABLE:
        logger.error("requests не встановлено. Виконайте: pip install requests")
        return None
    
    if not ollama_url:
        logger.error("Ollama URL не вказано")
        return None

    if model_name:
        model_name = model_name.strip() or DEFAULT_OLLAMA_MODEL
    else:
        model_name = DEFAULT_OLLAMA_MODEL
    ollama_url = _normalize_ollama_base_url(ollama_url)

    try:
        img_b64 = _encode_image_base64(image_path)
        if not img_b64:
            return None

        endpoint = f"{ollama_url}{OLLAMA_GENERATE_ENDPOINT}"

        payload = {
            "model": model_name,
            "prompt": _get_classify_prompt(),
            "images": [img_b64],
            "stream": False
        }

        response = requests.post(endpoint, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()
        response_text = result.get("response", "")

        return _parse_ai_response(response_text)
    except requests.exceptions.ConnectionError as exc:
        _log_ollama_issue_once(
            ("connection", ollama_url),
            "classify_with_ollama: Не вдалося підключитися до Ollama (%s): %s. "
            "Перевірте, що сервіс запущено командою 'ollama serve' і що %s доступний. "
            "Повідомлення користувачу: 'Не вдалося з'єднатися з локальною Ollama. "
            "Запустіть ollama serve та повторіть спробу.'",
            ollama_url,
            exc,
            endpoint,
        )
        return None
    except requests.exceptions.Timeout as exc:
        _log_ollama_issue_once(
            ("timeout", ollama_url),
            "classify_with_ollama: Час очікування Ollama вичерпано (%s): %s. "
            "Повідомлення користувачу: 'Ollama відповідає надто довго. "
            "Перевірте навантаження на модель або спробуйте ще раз.'",
            ollama_url,
            exc,
        )
        return None
    except requests.exceptions.HTTPError as exc:
        response = exc.response
        status_code = response.status_code if response is not None else "?"
        details = _get_ollama_error_details(response)
        details_lower = details.lower()

        if status_code == 404 and "model" in details_lower and "not found" in details_lower:
            _log_ollama_issue_once(
                ("model_not_found", ollama_url, model_name),
                "classify_with_ollama: Модель Ollama '%s' не знайдено на %s. "
                "Деталі відповіді: %s. Перевірте назву моделі через 'ollama list' "
                "і за потреби виконайте 'ollama pull %s'. "
                "Повідомлення користувачу: 'Модель %s не знайдено в Ollama. "
                "Звірте назву з ollama list або встановіть модель командою ollama pull %s.'",
                model_name,
                endpoint,
                details or exc,
                model_name,
                model_name,
                model_name,
            )
        elif status_code == 404:
            _log_ollama_issue_once(
                ("http_404", ollama_url),
                "classify_with_ollama: Ollama повернула 404 для %s. "
                "Можливі причини: сервіс не запущено, URL вказує не на Ollama, "
                "або використано неправильний endpoint. Деталі відповіді: %s. "
                "Повідомлення користувачу: 'Ollama відповіла 404. Перевірте URL, "
                "що працює саме сервер Ollama, і endpoint /api/generate.'",
                endpoint,
                details or exc,
            )
        else:
            _log_ollama_issue_once(
                ("http_error", ollama_url, str(status_code)),
                "classify_with_ollama: HTTP помилка Ollama (%s, статус %s): %s. "
                "Повідомлення користувачу: 'Ollama повернула HTTP %s. "
                "Перевірте лог сервісу та параметри запиту.'",
                endpoint,
                status_code,
                details or exc,
                status_code,
            )
        return None
    except Exception as exc:
        logger.warning(f"classify_with_ollama: {exc}")
        return None


# ---------------------------------------------------------------------------
# 6b. Ранжування набору зображень через Ollama (folder-level ordering)
# ---------------------------------------------------------------------------

_RANK_PROMPT_TEMPLATE = (
    "You are a professional e-commerce photo editor deciding the display order "
    "for {n} product photos on a marketplace.\n"
    "The photos are numbered 1 to {n} in the order they appear in the images array.\n\n"
    "IMPORTANT: Analyze each image carefully and reorder them optimally. "
    "Do NOT simply return [1, 2, 3, ...] - that would mean you didn't analyze them.\n\n"
    "=== STRICT CLASSIFICATION RULES ===\n\n"
    "IDEAL MAIN PHOTO (best candidate for position 1):\n"
    "  - Single product clearly visible\n"
    "  - Pure white or very light, clean background\n"
    "  - Absolutely NO text of any kind: no labels, no badges, no promotional text,\n"
    "    no watermarks, no brand name overlay, no price tags, no banners, no stickers\n"
    "  - No collages, no multiple products side by side, no extra decorative graphics\n"
    "  → ALWAYS place first if such a photo exists\n\n"
    "ALTERNATIVE MAIN PHOTO (second priority — needs additional editing):\n"
    "  - White or light background (mostly clean)\n"
    "  - BUT has any text, promotional labels, price badges, watermarks,\n"
    "    brand overlays, stickers, or graphic overlays of any kind\n"
    "  - Cannot be used as marketplace main image without processing to remove text\n"
    "  → Place after any ideal main photos, before gallery images\n\n"
    "GALLERY / ADDITIONAL PHOTOS (remaining positions in this order):\n"
    "  3. Packshots — product on solid/neutral non-white background, different angles\n"
    "  4. Detail shots — close-ups of materials, textures, product features\n"
    "  5. Lifestyle photos — product in use, with people, in real environment\n"
    "  6. Kit photos — product with packaging, box, all included accessories\n"
    "  7. Infographic photos — heavy text overlays, dimensions, technical specs, diagrams\n\n"
    "=== ORDERING RULES ===\n"
    "1. FIRST: Ideal main photo (pure white bg, absolutely no text, single product)\n"
    "2. THEN: Alternative main photos (white bg + any text/overlay — best quality first)\n"
    "3. THEN: Packshots on non-white backgrounds\n"
    "4. THEN: Detail shots (close-ups, textures, features)\n"
    "5. THEN: Lifestyle photos — smooth visual flow from product to usage context\n"
    "6. THEN: Kit photos (product with box and accessories)\n"
    "7. LAST: Infographic photos (more text = later position)\n\n"
    "Within each group: prefer cleaner, better-lit, higher-quality photos first.\n"
    "Images with LESS text should always come BEFORE images with more text.\n\n"
    "Respond with ONLY a JSON array of the photo numbers in your preferred order.\n"
    "All {n} numbers from 1 to {n} must appear exactly once.\n"
    "Example format: {example}\n"
    "Do NOT include any explanation — just the JSON array."
)

_RANK_GENERATION_OPTIONS = {
    # Мінімізуємо випадковість генерації, щоб порядок фото був повторюваним
    # між запусками на однаковому наборі зображень.
    # Трохи збільшено temperature та top_p для кращого аналізу зображень.
    "temperature": 0.1,
    "top_p": 0.3,
}


def _parse_rank_response(text: str, n_images: int) -> list[int] | None:
    """
    Парсить відповідь Ollama для ранжування набору зображень.

    Шукає JSON-масив цілих чисел у тексті відповіді.
    Повертає список 0-based індексів (перестановку range(n_images))
    або None при помилці.
    """
    # Шукаємо перший JSON-масив у відповіді
    match = re.search(r'\[[\d,\s]+\]', text.strip())
    if not match:
        logger.warning(
            "_parse_rank_response: JSON-масив не знайдено у відповіді: %r",
            text[:300],
        )
        return None

    try:
        arr = json.loads(match.group())
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("_parse_rank_response: помилка JSON-парсингу: %s", exc)
        return None

    if not isinstance(arr, list):
        logger.warning("_parse_rank_response: отримано не список: %r", type(arr))
        return None

    # Перевіряємо: всі елементи — цілі, рівно n_images штук, перестановка 1..n
    if not all(isinstance(x, int) for x in arr):
        logger.warning(
            "_parse_rank_response: масив містить не цілі числа: %r", arr[:20]
        )
        return None

    if sorted(arr) != list(range(1, n_images + 1)):
        logger.warning(
            "_parse_rank_response: неповна або невалідна перестановка: %r "
            "(очікувалось 1..%d)",
            arr,
            n_images,
        )
        return None

    # Перетворюємо 1-based → 0-based
    return [x - 1 for x in arr]


def rank_images_with_ollama(
    image_paths: list[str],
    ollama_url: str | None,
    model_name: str = DEFAULT_OLLAMA_MODEL,
) -> list[int] | None:
    """
    Ранжує набір зображень продукту через Ollama.

    Надсилає всі зображення (до OLLAMA_MAX_IMAGES_PER_RANK) одним запитом і
    просить vision-модель визначити оптимальний порядок їх відображення.

    Параметри:
      image_paths — список шляхів до зображень (мінімум 2, не більше
                    OLLAMA_MAX_IMAGES_PER_RANK)
      ollama_url  — базовий URL Ollama API (наприклад, http://localhost:11434)
      model_name  — назва vision-моделі в Ollama (наприклад, "llava")

    Повертає список 0-based індексів — перестановку range(len(image_paths)) —
    де елемент з індексом 0 вказує на зображення, що має бути першим, і т.д.
    Повертає None при будь-якій помилці; у цьому випадку слід використовувати
    стандартне категорійне сортування.

    Примітка: Модель повинна підтримувати обробку зображень (vision-модель).
    """
    if not REQUESTS_AVAILABLE:
        logger.error("requests не встановлено. Виконайте: pip install requests")
        return None

    if not ollama_url:
        logger.error("rank_images_with_ollama: Ollama URL не вказано")
        return None

    n = len(image_paths)
    if n < 2:
        logger.info(
            "rank_images_with_ollama: менше 2 зображень (%d) — ранжування не потрібне",
            n,
        )
        return None

    if n > OLLAMA_MAX_IMAGES_PER_RANK:
        logger.warning(
            "rank_images_with_ollama: кількість зображень (%d) перевищує ліміт (%d). "
            "Ранжування Ollama пропущено — буде використано стандартне сортування.",
            n,
            OLLAMA_MAX_IMAGES_PER_RANK,
        )
        return None

    model_name = (model_name or "").strip() or DEFAULT_OLLAMA_MODEL

    ollama_url = _normalize_ollama_base_url(ollama_url)

    # Кодуємо всі зображення
    images_b64: list[str] = []
    for path in image_paths:
        b64 = _encode_image_base64(path)
        if b64 is None:
            logger.warning(
                "rank_images_with_ollama: не вдалося кодувати %s — ранжування пропущено",
                path,
            )
            return None
        images_b64.append(b64)

    # Будуємо приклад відповіді для промпту.
    # Використовуємо різні перестановки залежно від кількості зображень,
    # щоб показати модель, що потрібно РЕАЛЬНО аналізувати, а не копіювати [1,2,3,...]
    if n == 2:
        example = [2, 1]
    elif n == 3:
        example = [1, 3, 2]
    elif n == 4:
        example = [3, 1, 2, 4]
    elif n == 5:
        example = [2, 4, 1, 3, 5]
    elif n == 6:
        example = [5, 6, 4, 1, 2, 3]
    else:
        # Для більших наборів - просто показуємо, що порядок має бути змінений
        example = list(range(2, min(6, n+1))) + [1] + list(range(6, n+1))

    prompt = _get_rank_prompt_template().format(n=n, example=example)
    endpoint = f"{ollama_url}{OLLAMA_GENERATE_ENDPOINT}"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": images_b64,
        "stream": False,
        "options": _RANK_GENERATION_OPTIONS,
    }

    try:
        response = requests.post(endpoint, json=payload, timeout=120)
        response.raise_for_status()

        result = response.json()
        response_text = result.get("response", "")
        logger.info(
            "rank_images_with_ollama: відповідь моделі: %r",
            response_text[:300],
        )

        indices = _parse_rank_response(response_text, n)
        if indices is None:
            logger.warning(
                "rank_images_with_ollama: не вдалося розпарсити відповідь. "
                "Відповідь: %r. Буде використано стандартне сортування.",
                response_text[:500]
            )
        else:
            logger.info(
                "rank_images_with_ollama: успішне ранжування. Порядок: %s",
                [i+1 for i in indices]  # Конвертуємо назад до 1-based для читабельності
            )
        return indices

    except requests.exceptions.ConnectionError as exc:
        _log_ollama_issue_once(
            ("rank_connection", ollama_url),
            "rank_images_with_ollama: Не вдалося підключитися до Ollama (%s): %s. "
            "Буде використано стандартне сортування.",
            ollama_url,
            exc,
        )
        return None
    except requests.exceptions.Timeout as exc:
        _log_ollama_issue_once(
            ("rank_timeout", ollama_url),
            "rank_images_with_ollama: Час очікування Ollama вичерпано (%s): %s. "
            "Буде використано стандартне сортування.",
            ollama_url,
            exc,
        )
        return None
    except requests.exceptions.HTTPError as exc:
        response = exc.response
        status_code = response.status_code if response is not None else "?"
        details = _get_ollama_error_details(response)
        _log_ollama_issue_once(
            ("rank_http_error", ollama_url, str(status_code)),
            "rank_images_with_ollama: HTTP помилка Ollama (%s, статус %s): %s. "
            "Буде використано стандартне сортування.",
            endpoint,
            status_code,
            details or exc,
        )
        return None
    except Exception as exc:
        logger.warning("rank_images_with_ollama: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 7. Класифікація через локальну CLIP модель
# ---------------------------------------------------------------------------

_CLIP_MODEL_CACHE: dict = {}  # {device: (model, preprocess)}

_CLIP_TEXT_PROMPTS = [
    "product photo on white or very light background, clear product showcase with good visibility, well-lit professional product photography, main product focus",
    "product packshot on solid neutral background, different angle",
    "close-up macro photo of product detail or material texture",
    "product placed in interior living space, lifestyle photo",
    "product with its packaging box and all accessories, kit contents",
    "product infographic with text labels, dimensions and technical specs",
]


def classify_with_clip(image_path: str) -> str | None:
    """
    Класифікує зображення за допомогою локальної CLIP моделі (без API).

    При першому запуску завантажує модель ViT-B/32 (~350 MB).
    Потрібні: pip install torch torchvision
              pip install git+https://github.com/openai/CLIP.git

    Повертає рядок категорії або None при помилці.
    """
    if not CLIP_AVAILABLE:
        logger.error("torch/CLIP не встановлено.")
        return None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if device not in _CLIP_MODEL_CACHE:
            # ---- Модель: змініть 'ViT-B/32' на 'ViT-L/14' для кращої точності ----
            model, preprocess = clip_module.load("ViT-B/32", device=device)
            _CLIP_MODEL_CACHE[device] = (model, preprocess)
        else:
            model, preprocess = _CLIP_MODEL_CACHE[device]

        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        text_tokens = clip_module.tokenize(_CLIP_TEXT_PROMPTS).to(device)

        with torch.no_grad():
            logits, _ = model(image, text_tokens)
            probs = logits.softmax(dim=-1).cpu().numpy()[0]

        return CATEGORY_ORDER[int(np.argmax(probs))]

    except Exception as exc:
        logger.warning("classify_with_clip: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 7. Основна функція класифікації
# ---------------------------------------------------------------------------

def classify_image(image_path: str,
                   api_type: str = "none",
                   api_key: str | None = None,
                   ollama_url: str | None = None,
                   ollama_model: str = DEFAULT_OLLAMA_MODEL,
                   white_bg_score: float = 0.0,
                   has_text: bool = False,
                   detected_text: str = "") -> tuple[str, float, str]:
    """
    Визначає категорію зображення.

    Параметри:
      api_type — метод аналізу: 'gemini' | 'openai' | 'ollama' | 'clip' | 'none'
      api_key  — API ключ (для gemini/openai)
      ollama_url — URL Ollama API (для ollama)
      ollama_model — назва моделі в Ollama (для ollama)
      white_bg_score, has_text, detected_text — результати OpenCV/OCR аналізу

    Повертає (category, confidence, method_used).
    """
    ai_category: str | None = None
    method = "opencv"

    # --- Спроба AI класифікації ---
    if api_type == "gemini" and api_key:
        ai_category = classify_with_gemini(image_path, api_key)
        if ai_category:
            method = "gemini"
    elif api_type == "openai" and api_key:
        ai_category = classify_with_openai(image_path, api_key)
        if ai_category:
            method = "openai"
    elif api_type == "ollama" and ollama_url:
        ai_category = classify_with_ollama(image_path, ollama_url, ollama_model)
        if ai_category:
            method = f"ollama:{ollama_model}"
    elif api_type == "clip":
        ai_category = classify_with_clip(image_path)
        if ai_category:
            method = "clip"

    # --- Поєднання AI результату з OpenCV/OCR ---
    if ai_category:
        # Якщо AI каже "main", але OCR виявив будь-який текст — понижуємо до packshot.
        # Наявність тексту, плашок або водяних знаків є блокуючою умовою для
        # "ідеального головного фото". Такі фото класифікуються як альтернативні.
        if ai_category == CATEGORY_MAIN and has_text:
            ai_category = CATEGORY_PACKSHOT
        return ai_category, 1.0, method

    # --- Резервна класифікація лише на основі OpenCV + OCR ---
    word_count = len(detected_text.split()) if detected_text else 0

    if has_text:
        if word_count > MAX_WORDS_FOR_MAIN_CATEGORY:
            # Багато тексту → скоріше за все інфографіка
            return CATEGORY_INFOGRAPHIC, 0.6, method
        else:
            return CATEGORY_PACKSHOT, 0.5, method

    # Налаштування порогів:
    #   WHITE_BG_HIGH   — поріг для ідеального білого фону (main)
    #   WHITE_BG_MEDIUM — поріг для пекшоту на світлому фоні
    WHITE_BG_HIGH = 0.85
    WHITE_BG_MEDIUM = 0.55

    if white_bg_score >= WHITE_BG_HIGH:
        return CATEGORY_MAIN, white_bg_score, method
    elif white_bg_score >= WHITE_BG_MEDIUM:
        return CATEGORY_PACKSHOT, white_bg_score, method
    else:
        return CATEGORY_LIFESTYLE, 1.0 - white_bg_score, method


# ---------------------------------------------------------------------------
# 8. Повний аналіз одного зображення
# ---------------------------------------------------------------------------

def analyze_image(image_path: str,
                  api_type: str = "none",
                  api_key: str | None = None,
                  ollama_url: str | None = None,
                  ollama_model: str = DEFAULT_OLLAMA_MODEL) -> dict:
    """
    Виконує повний аналіз зображення і повертає словник з результатами.

    Результуючий словник:
      path           — абсолютний шлях до файлу
      filename       — ім'я файлу
      category       — категорія (main/packshot/detail/lifestyle/kit/infographic)
      white_bg_score — оцінка білизни фону (0.0–1.0)
      has_text       — чи виявлено текст
      detected_text  — знайдений текст
      method         — метод класифікації
      error          — рядок помилки або None
    """
    result: dict = {
        "path": image_path,
        "filename": os.path.basename(image_path),
        "category": CATEGORY_PACKSHOT,
        "white_bg_score": 0.0,
        "has_text": False,
        "detected_text": "",
        "method": "opencv",
        "error": None,
    }

    try:
        if not os.path.exists(image_path):
            result["error"] = "Файл не знайдено"
            return result

        # Перевірка цілісності файлу
        with Image.open(image_path) as img:
            img.verify()

        result["white_bg_score"] = detect_white_background(image_path)
        result["has_text"], result["detected_text"] = detect_text_or_watermarks(image_path)

        category, confidence, method = classify_image(
            image_path,
            api_type=api_type,
            api_key=api_key,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
            white_bg_score=result["white_bg_score"],
            has_text=result["has_text"],
            detected_text=result["detected_text"],
        )
        result["category"] = category
        result["confidence"] = confidence
        result["method"] = method

    except Exception as exc:
        result["error"] = str(exc)

    return result
