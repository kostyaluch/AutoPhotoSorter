"""
sorter.py — Логіка сортування та перейменування фотографій для AutoPhotoSorter.

Пріоритет сортування:
  1. main        — Головне фото (позиція 1)
  2. packshot    — Пекшоти на однотонному фоні (позиції 2-N)
  3. detail      — Деталі та макрозйомка
  4. lifestyle   — Фото в інтер'єрі / lifestyle
  5. kit         — Комплектація
  6. infographic — Інфографіка з розмірами та характеристиками
"""

import os
import shutil
import logging

from analyzer import (
    analyze_image,
    rank_images_with_ollama,
    CATEGORY_ORDER,
    CATEGORY_MAIN,
    CATEGORY_PACKSHOT,
    DEFAULT_OLLAMA_MODEL,
    IMAGE_EXTENSIONS,
    OLLAMA_MAX_IMAGES_PER_RANK,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Допоміжні функції
# ---------------------------------------------------------------------------

def get_images_in_folder(folder_path: str) -> list[str]:
    """
    Повертає відсортований список шляхів до всіх зображень у папці.
    Ігнорує підпапки.
    """
    images = []
    try:
        for filename in os.listdir(folder_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                images.append(os.path.join(folder_path, filename))
    except OSError as exc:
        logger.warning("get_images_in_folder(%s): %s", folder_path, exc)
    return sorted(images)


def find_subfolders_with_images(root_dir: str) -> list[str]:
    """
    Рекурсивно знаходить всі папки, які містять зображення.
    Ігнорує папки, назва яких починається з '_' або '.'.

    Повертає відсортований список шляхів до папок.
    """
    result = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        # Прибираємо приховані/тимчасові папки зі списку обходу
        dirnames[:] = [
            d for d in sorted(dirnames)
            if not d.startswith(".") and not d.startswith("_")
        ]
        if get_images_in_folder(dirpath):
            result.append(dirpath)
    return result


# ---------------------------------------------------------------------------
# Сортування зображень за категоріями
# ---------------------------------------------------------------------------

def sort_images(images_data: list[dict]) -> tuple[list[dict], dict[str, list[dict]]]:
    """
    Сортує список результатів аналізу за пріоритетом категорій.

    Всередині категорії 'main' впорядковує за оцінкою білизни фону (спадання).
    Всередині інших категорій зберігає оригінальний алфавітний порядок.

    Повертає (sorted_list, categorized_dict).
    """
    categorized: dict[str, list[dict]] = {cat: [] for cat in CATEGORY_ORDER}

    for img_data in images_data:
        cat = img_data.get("category", CATEGORY_PACKSHOT)
        if cat not in categorized:
            cat = CATEGORY_PACKSHOT
        categorized[cat].append(img_data)

    # Найкраще головне фото — перше
    categorized[CATEGORY_MAIN].sort(
        key=lambda x: x.get("white_bg_score", 0.0), reverse=True
    )

    sorted_list: list[dict] = []
    for cat in CATEGORY_ORDER:
        sorted_list.extend(categorized[cat])

    return sorted_list, categorized


def find_best_fallback(images_data: list[dict]) -> dict | None:
    """
    Вибирає найкраще фото для позиції 1 коли немає ідеального головного.

    Критерій: найвища оцінка білизни фону (найкраще видно товар, найізольованіший).
    """
    valid = [img for img in images_data if not img.get("error")]
    if not valid:
        return None
    return max(valid, key=lambda x: x.get("white_bg_score", 0.0))


# ---------------------------------------------------------------------------
# Обробка однієї папки
# ---------------------------------------------------------------------------

def process_folder(folder_path: str,
                   api_type: str = "none",
                   api_key: str | None = None,
                   ollama_url: str | None = None,
                   ollama_model: str = DEFAULT_OLLAMA_MODEL,
                   progress_callback=None) -> dict:
    """
    Аналізує та сортує всі зображення в папці.

    Параметри:
      folder_path       — шлях до папки з зображеннями
      api_type          — метод AI: 'gemini' | 'openai' | 'ollama' | 'clip' | 'none'
      api_key           — API ключ (для gemini/openai)
      ollama_url        — URL Ollama API (для ollama)
      ollama_model      — назва моделі в Ollama (для ollama)
      progress_callback — функція(current, total, message) для відображення прогресу

    Результат (dict):
      folder            — шлях до папки
      folder_name       — назва папки
      sorted_images     — список проаналізованих зображень у порядку сортування
      has_ideal_main    — bool: чи знайдено ідеальне головне фото
      fallback_used     — bool: чи використано альтернативу
      fallback_image    — dict даних про альтернативне головне фото
      renamed_files     — заповнюється після виклику rename_images_in_folder()
      error             — рядок помилки або None
    """
    result: dict = {
        "folder": folder_path,
        "folder_name": os.path.basename(folder_path),
        "sorted_images": [],
        "has_ideal_main": False,
        "has_alternative_main": False,
        "alternative_main_image": None,
        "fallback_used": False,
        "fallback_image": None,
        "renamed_files": [],
        "ollama_ranked": False,
        "error": None,
    }

    try:
        images = get_images_in_folder(folder_path)

        if not images:
            result["error"] = "No images found"
            return result

        # --- Аналіз кожного зображення ---
        images_data: list[dict] = []
        for i, img_path in enumerate(images):
            if progress_callback:
                progress_callback(i, len(images),
                                  f"Аналіз: {os.path.basename(img_path)}")
            img_data = analyze_image(img_path, api_type=api_type, api_key=api_key, 
                                    ollama_url=ollama_url, ollama_model=ollama_model)
            images_data.append(img_data)

        # --- Сортування ---
        sorted_images, categorized = sort_images(images_data)
        result["sorted_images"] = sorted_images

        # --- Визначення статусу головного фото ---
        main_photos = categorized[CATEGORY_MAIN]
        packshot_photos = categorized[CATEGORY_PACKSHOT]

        if main_photos:
            best_main = main_photos[0]
            # Ідеальне головне: немає тексту/плашок/водяних знаків.
            # Після виправлення classify_image це гарантовано для фото, що пройшли
            # OCR-перевірку, але перевіряємо ще раз як запобіжний захід.
            if not best_main.get("has_text", False):
                result["has_ideal_main"] = True
            else:
                # Edge case: AI класифікував як main, але OCR бачить текст
                result["has_alternative_main"] = True
                result["alternative_main_image"] = best_main

        # Шукаємо "альтернативне головне фото" серед пекшотів:
        # фото на білому/світлому фоні з текстом/плашками — потребує обробки
        if not result["has_ideal_main"] and not result["has_alternative_main"]:
            ALTERNATIVE_MAIN_THRESHOLD = 0.55
            alt_candidates = [
                p for p in packshot_photos
                if p.get("white_bg_score", 0.0) >= ALTERNATIVE_MAIN_THRESHOLD
                and p.get("has_text", False)
            ]
            if alt_candidates:
                best_alt = max(
                    alt_candidates,
                    key=lambda x: x.get("white_bg_score", 0.0)
                )
                result["has_alternative_main"] = True
                result["alternative_main_image"] = best_alt

        if not result["has_ideal_main"] and not result["has_alternative_main"]:
            # Жодного придатного головного фото — використовуємо найкраще доступне
            result["fallback_used"] = True
            fallback = find_best_fallback(images_data)
            if fallback:
                result["fallback_image"] = fallback
                # Ставимо fallback на перше місце
                sorted_images = [
                    img for img in sorted_images
                    if img["path"] != fallback["path"]
                ]
                sorted_images.insert(0, fallback)
                result["sorted_images"] = sorted_images
        elif result["has_alternative_main"] and result["alternative_main_image"]:
            # Переконуємось, що альтернативне головне фото стоїть першим
            alt_img = result["alternative_main_image"]
            if sorted_images and sorted_images[0]["path"] != alt_img["path"]:
                sorted_images = [
                    img for img in sorted_images
                    if img["path"] != alt_img["path"]
                ]
                sorted_images.insert(0, alt_img)
                result["sorted_images"] = sorted_images

        # --- Ollama folder-level ranking (overrides category-based sort) ---
        # When Ollama is selected, send all images together so the model can
        # evaluate the full product set and decide the optimal display order.
        if api_type == "ollama" and ollama_url:
            n = len(images_data)
            logger.info(
                "process_folder(%s): Ollama вибрано, %d зображень для ранжування",
                os.path.basename(folder_path),
                n
            )
            if 2 <= n <= OLLAMA_MAX_IMAGES_PER_RANK:
                if progress_callback:
                    progress_callback(
                        n, n,
                        f"Ollama: ранжування {n} фото разом…"
                    )
                all_paths = [img["path"] for img in images_data]
                rank_order = rank_images_with_ollama(
                    all_paths, ollama_url, ollama_model
                )
                if rank_order is not None:
                    result["sorted_images"] = [images_data[i] for i in rank_order]
                    result["ollama_ranked"] = True
                    logger.info(
                        "process_folder(%s): Ollama ранжування застосовано",
                        os.path.basename(folder_path),
                    )

                    # Оновлюємо статус головного фото на основі вибору Ollama:
                    # перевіряємо перше фото у відсортованому Ollama списку
                    first_photo = result["sorted_images"][0] if result["sorted_images"] else None
                    if first_photo:
                        first_has_text = first_photo.get("has_text", False)
                        first_white_bg = first_photo.get("white_bg_score", 0.0)
                        # Скидаємо попередні прапорці і визначаємо наново
                        result["has_ideal_main"] = False
                        result["has_alternative_main"] = False
                        result["alternative_main_image"] = None
                        result["fallback_used"] = False
                        result["fallback_image"] = None

                        IDEAL_BG_THRESHOLD = 0.65
                        ALTERNATIVE_BG_THRESHOLD = 0.45
                        if not first_has_text and first_white_bg >= IDEAL_BG_THRESHOLD:
                            result["has_ideal_main"] = True
                        elif first_has_text and first_white_bg >= ALTERNATIVE_BG_THRESHOLD:
                            result["has_alternative_main"] = True
                            result["alternative_main_image"] = first_photo
                        # else: Ollama помістила нейтральне фото на перше місце
                else:
                    logger.info(
                        "process_folder(%s): Ollama ранжування не вдалося — "
                        "використовується стандартне сортування за категоріями",
                        os.path.basename(folder_path),
                    )
            elif n > OLLAMA_MAX_IMAGES_PER_RANK:
                logger.warning(
                    "process_folder(%s): %d зображень перевищує ліміт "
                    "ранжування Ollama (%d) — використовується стандартне сортування",
                    os.path.basename(folder_path),
                    n,
                    OLLAMA_MAX_IMAGES_PER_RANK,
                )

    except Exception as exc:
        result["error"] = str(exc)
        logger.exception("process_folder(%s)", folder_path)

    return result


# ---------------------------------------------------------------------------
# Перейменування файлів
# ---------------------------------------------------------------------------

def rename_images_in_folder(folder_path: str,
                             sorted_images: list[dict],
                             dry_run: bool = False) -> list[dict]:
    """
    Перейменовує зображення у папці згідно з відсортованим списком.

    Схема іменування: 01.jpg, 02.jpg, … (максимум 99 файлів).
    Нормалізує .jpeg → .jpg, .tiff → .tif.

    Параметри:
      dry_run — якщо True, файли НЕ перейменовуються (лише повертається план).

    Повертає список словників:
      { old_path, old_name, new_path, new_name, category }
    """
    renames: list[dict] = []

    # Відбираємо лише валідні зображення (без помилок)
    valid = [img for img in sorted_images if not img.get("error")]

    # Максимальна кількість файлів визначається двозначною схемою іменування:
    # 01.jpg … 99.jpg → не більше 99 позицій.
    MAX_FILES = 99
    valid = valid[:MAX_FILES]

    for idx, img_data in enumerate(valid, start=1):
        old_path = img_data["path"]
        ext = os.path.splitext(old_path)[1].lower()

        # Нормалізація розширення
        ext = {".jpeg": ".jpg", ".tiff": ".tif"}.get(ext, ext)

        new_filename = f"{idx:02d}{ext}"
        new_path = os.path.join(folder_path, new_filename)

        renames.append({
            "old_path": old_path,
            "old_name": os.path.basename(old_path),
            "new_path": new_path,
            "new_name": new_filename,
            "category": img_data.get("category", "unknown"),
        })

    if dry_run:
        return renames

    # --- Двоетапне перейменування через тимчасову папку ---
    # щоб уникнути конфліктів (наприклад: 01.jpg вже існує і є цільовою назвою)
    temp_dir = os.path.join(folder_path, "_autosort_tmp_")
    try:
        os.makedirs(temp_dir, exist_ok=True)

        # Крок 1: копіюємо в тимчасову папку під новими іменами
        tmp_files: list[tuple[str, str]] = []
        for rename in renames:
            tmp_path = os.path.join(temp_dir, rename["new_name"])
            shutil.copy2(rename["old_path"], tmp_path)
            tmp_files.append((tmp_path, rename["new_path"]))

        # Крок 2: видаляємо оригінали
        original_paths = {r["old_path"] for r in renames}
        for orig in original_paths:
            try:
                os.remove(orig)
            except OSError as exc:
                logger.warning("Не вдалося видалити %s: %s", orig, exc)

        # Крок 3: переміщуємо з тимчасової папки до цільової
        for tmp_path, dest_path in tmp_files:
            shutil.move(tmp_path, dest_path)

    finally:
        # Завжди прибираємо тимчасову папку
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    return renames
