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

import os
import base64
import logging
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
        response = model.generate_content([_CLASSIFY_PROMPT, img])
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
                    {"type": "text", "text": _CLASSIFY_PROMPT},
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

def classify_with_ollama(image_path: str, ollama_url: str, model_name: str = "llava") -> str | None:
    """
    Класифікує зображення через Ollama (локальні моделі).

    Налаштування:
      ollama_url — URL Ollama API (наприклад, http://localhost:11434)
      model_name — назва моделі в Ollama (наприклад, "llava", "gemma", "bakllava")

    Для використання моделі Gemma з візуальними можливостями рекомендується
    використати llava або bakllava, оскільки чиста Gemma не підтримує зображення.

    Повертає рядок категорії або None при помилці.
    """
    if not REQUESTS_AVAILABLE:
        logger.error("requests не встановлено. Виконайте: pip install requests")
        return None
    try:
        img_b64 = _encode_image_base64(image_path)
        if not img_b64:
            return None

        # Видаляємо trailing slash з URL
        ollama_url = ollama_url.rstrip('/')
        endpoint = f"{ollama_url}/api/generate"
        
        payload = {
            "model": model_name,
            "prompt": _CLASSIFY_PROMPT,
            "images": [img_b64],
            "stream": False
        }
        
        response = requests.post(endpoint, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        response_text = result.get("response", "")
        
        return _parse_ai_response(response_text)
    except Exception as exc:
        logger.warning("classify_with_ollama: %s", exc)
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
                   ollama_model: str = "llava",
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
        # Якщо AI каже "main", але OCR знайшов багато тексту — понижуємо до packshot
        # Малі логотипи або брендування (до MAX_WORDS_FOR_MAIN_CATEGORY слів) приймаються
        if ai_category == CATEGORY_MAIN and has_text and detected_text:
            word_count = len(detected_text.split())
            if word_count > MAX_WORDS_FOR_MAIN_CATEGORY:
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
                  ollama_model: str = "llava") -> dict:
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
