# AutoPhotoSorter 🗂️

**Програма для автоматичного аналізу, сортування та перейменування фотографій товарів.**

Обходить папки з фотографіями артикулів, аналізує кожне зображення та перейменовує їх від `01.jpg` до `NN.jpg` за чіткою пріоритетністю. Формує Excel-звіт зі статусами всіх папок.

---

## Можливості

| Функція | Опис |
|---|---|
| 🖼️ Визначення білого фону | OpenCV аналізує яскравість країв та загальну яскравість зображення |
| 🔤 Виявлення тексту / водяних знаків | OCR (Tesseract) — опціонально |
| 🤖 AI класифікація контенту | Google Gemini API / OpenAI Vision API / Ollama (llava, bakllava) / локальна CLIP модель |
| 📊 Excel-звіт | Два аркуші: зведений звіт та деталі по кожному файлу |
| 🗂️ Масова обробка | Рекурсивна обробка всіх підпапок вхідної директорії |
| 🔍 Тестовий режим | Перегляд результатів без фактичного перейменування файлів |

---

## Пріоритет сортування

| Позиція | Категорія | Опис |
|---|---|---|
| **1** | `main` | Головне фото: білий/світлий фон (не обов'язково ідеально чистий), може бути присутній невеликий логотип/брендування (до 6 слів тексту), але це має бути саме фото товару |
| **2–N** | `packshot` | Інші ракурси товару на однотонному фоні |
| **Далі** | `detail` | Деталі, макрозйомка матеріалів |
| **Далі** | `lifestyle` | Товар в інтер'єрі / реальне використання |
| **Далі** | `kit` | Комплектація: коробка, аксесуари |
| **Останні** | `infographic` | Інфографіка: розміри, характеристики, схеми |

Якщо ідеального головного фото не знайдено — обирається найбільш підходяще зображення на основі оцінки білизни фону (найізольованіший товар), а папка позначається у звіті для ручної перевірки.

---

## Встановлення

### 1. Клонування репозиторію

```bash
git clone https://github.com/kostyaluch/AutoPhotoSorter.git
cd AutoPhotoSorter
```

### 2. Створення та активація віртуального середовища (рекомендовано)

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

### 3. Встановлення залежностей

```bash
pip install -r requirements.txt
```

> **Мінімально необхідні пакети** (якщо не потрібен AI та OCR):
> ```bash
> pip install opencv-python Pillow numpy openpyxl
> ```

---

## Опціональні залежності

### OCR (виявлення тексту та водяних знаків)

1. Встановіть системний пакет **Tesseract**:
   - **Windows**: [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
   - **Linux**: `sudo apt-get install tesseract-ocr`
   - **macOS**: `brew install tesseract`
2. Потім встановіть Python-обгортку: `pip install pytesseract`

### Google Gemini API

```bash
pip install google-generativeai
```
Отримайте ключ на [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

### OpenAI Vision API

```bash
pip install openai
```
Отримайте ключ на [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)

### Ollama (локальні моделі, локальний API)

Ollama дозволяє запускати моделі локально й використовувати їх через HTTP API. У цьому проєкті запит іде на endpoint `POST /api/generate`, а в GUI потрібно вказувати **базовий URL сервера** — за замовчуванням `http://localhost:11434`.

> **Важливо:** для аналізу фотографій потрібна **vision-модель**. Текстові моделі на кшталт `gemma4:latest` можна використати для перевірки самого API, але вони не підходять для класифікації зображень у цьому застосунку.

#### 1. Встановлення Ollama

- **Windows / macOS**: завантажте інсталятор з [https://ollama.com/download](https://ollama.com/download)
- **Linux**:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

#### 2. Запуск сервісу Ollama

На деяких системах Ollama запускається у фоні автоматично після встановлення. Якщо ні — стартуйте сервер вручну:

```bash
ollama serve
```

Після запуску сервер має слухати:

```text
http://localhost:11434
```

#### 3. Перевірка, що модель встановлена локально

Подивіться список доступних моделей:

```bash
ollama list
```

Приклад:

```text
NAME             ID           SIZE
llava:latest     ...          ...
gemma4:latest    ...          ...
```

Якщо потрібної vision-моделі немає — встановіть її:

```bash
ollama pull llava
# або
ollama pull bakllava
```

#### 4. Перевірка доступності Ollama через curl

##### Linux / macOS / Git Bash

```bash
curl http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llava",
    "prompt": "Привіт",
    "stream": false
  }'
```

##### Windows Command Prompt (cmd.exe)

У `cmd.exe` JSON краще передавати **в один рядок**:

```bat
curl http://localhost:11434/api/generate -H "Content-Type: application/json" -d "{\"model\":\"gemma4:latest\",\"prompt\":\"Привіт\",\"stream\":false}"
```

Саме тут часто трапляється помилка, як на вашому скриншоті: `curl` запускається, а наступні рядки JSON `cmd.exe` намагається виконати як окремі команди (`"model"` is not recognized...). Це проблема формату команди Windows, а не обов'язково проблема в Ollama.

<img src="https://github.com/user-attachments/assets/b8569ce1-4f60-42b1-9f6f-d26ffa93682f" alt="Приклад перевірки Ollama у Windows CMD" width="900">

##### Швидка перевірка списку локальних моделей через API

```bash
curl http://localhost:11434/api/tags
```

#### 5. Перевірка Ollama з Python

Мінімальний приклад для перевірки API:

```python
import requests

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "gemma4:latest"  # або точна назва з `ollama list`

payload = {
    "model": MODEL_NAME,
    "prompt": "Відповідай одним словом: працює",
    "stream": False,
}

response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60)
response.raise_for_status()
print(response.json())
```

Якщо хочете перевірити саме **робочий сценарій цього проєкту**, використовуйте модель із підтримкою зображень:

```python
import base64
import requests

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "llava"  # або інша vision-модель з `ollama list`
IMAGE_PATH = "example.jpg"

with open(IMAGE_PATH, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "model": MODEL_NAME,
    "prompt": "Опиши фото одним коротким реченням українською.",
    "images": [image_b64],
    "stream": False,
}

response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60)
response.raise_for_status()
print(response.json()["response"])
```

#### 6. Правильне підключення Ollama в AutoPhotoSorter

1. Запустіть сервер:
   ```bash
   ollama serve
   ```
2. Переконайтеся, що модель є у списку:
   ```bash
   ollama list
   ```
3. У GUI програми:
   - виберіть **"Ollama (локальна модель)"**
   - у полі **Ollama URL** вкажіть базову адресу, наприклад `http://localhost:11434`
   - у полі **Ollama модель** вкажіть **точну назву** з `ollama list`
4. За потреби можна задати дефолтні значення через змінні середовища:
   ```bash
   AUTOPHOTOSORTER_OLLAMA_URL=http://localhost:11434
   AUTOPHOTOSORTER_OLLAMA_MODEL=llava
   ```

> Порада: якщо випадково вставити в GUI `http://localhost:11434/api` або навіть `http://localhost:11434/api/generate`, програма тепер нормалізує адресу до базового URL. Але найкраще все одно вказувати саме `http://localhost:11434`.

### Локальна CLIP модель (без API, ~350 MB)

```bash
# PyTorch (CPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CLIP:
pip install git+https://github.com/openai/CLIP.git
```

---

## Запуск

```bash
python main.py
```

Відкриється графічний інтерфейс:

1. **Виберіть вхідну папку** — батьківська папка, яка містить підпапки з фотографіями товарів.
2. **Вкажіть шлях для збереження звіту** (`.xlsx`).
3. **Оберіть метод аналізу**:
   - *Тільки OpenCV* — без API, найшвидший варіант
   - *Google Gemini / OpenAI* — вимагає API ключ, найточніший
   - *Ollama* — локальні моделі (llava, bakllava), не потребує API ключа, працює офлайн
   - *Локальна CLIP* — без API, але повільніше та вимагає ~350 MB
4. За потреби увімкніть **Тестовий режим** (файли не перейменовуватимуться).
5. Натисніть **▶ Почати обробку**.

Після завершення відкриється звіт у Excel.

---

## Структура проєкту

```
AutoPhotoSorter/
├── main.py           # GUI-застосунок (tkinter)
├── analyzer.py       # Аналіз зображень (OpenCV, OCR, AI API)
├── sorter.py         # Логіка сортування та перейменування
├── reporter.py       # Генерація Excel-звіту (openpyxl)
├── requirements.txt  # Список залежностей
└── README.md         # Ця інструкція
```

---

## Структура звіту Excel

**Аркуш 1 — «Зведений звіт»:**

| Папка | Статус | Кількість фото | Головне фото | Метод аналізу | Перейменовано | Примітки |
|---|---|---|---|---|---|---|
| Артикул_001 | ✅ Успішно | 8 | ✅ Ідеальне знайдено | gemini | 8 | |
| Артикул_002 | ⚠️ Потрібна перевірка | 5 | ⚠️ Альтернатива: IMG_003.jpg | opencv | 5 | Ідеальне головне фото не знайдено… |

**Аркуш 2 — «Деталі файлів»:** розбивка по кожному зображенню з категорією, оцінкою фону та новою назвою.

---

## Налаштування чутливості (OpenCV)

У файлі `analyzer.py` можна змінити пороги:

```python
# У функції detect_white_background():
pixel_threshold = 240   # Поріг яскравості "білого" пікселя (0–255).
                        # Зменшіть до 220 для прийняття кремово-білих фонів.
border_fraction = 0.10  # Частка краю зображення для аналізу (0–0.5).

# У функції classify_image():
WHITE_BG_HIGH   = 0.85  # Поріг для категорії "main" (ідеально білий фон)
WHITE_BG_MEDIUM = 0.55  # Поріг для категорії "packshot" (світлий фон)
```

---

## Олія для Ollama: часті питання та рішення

### Чому бачу `404 Client Error: Not Found for url: http://localhost:11434/api/generate`?

Найчастіші причини:

1. **Ollama не запущена**
   - Запустіть:
     ```bash
     ollama serve
     ```
   - Потім перевірте:
     ```bash
     curl http://localhost:11434/api/tags
     ```

2. **На порту `11434` відповідає не Ollama**
   - Перевірте, що відкривається саме Ollama API, а не інший локальний сервіс.

3. **Вказано неправильний URL або endpoint**
   - У програмі задавайте базовий URL: `http://localhost:11434`
   - Не потрібно вручну дописувати `/api/generate`

4. **Модель не встановлена або неправильно названа**
   - Перевірте:
     ```bash
     ollama list
     ```
   - Якщо моделі немає:
     ```bash
     ollama pull llava
     ```

5. **Модель є, але вона не підтримує зображення**
   - `gemma4:latest` підходить для текстових тестів API
   - для цього проєкту потрібна vision-модель: `llava`, `bakllava` або інша сумісна мультимодальна модель

6. **Проблема саме в Windows `curl`, а не в Ollama**
   - У `cmd.exe` JSON має бути в один рядок або з правильно екранованими лапками
   - Якщо є сумніви — перевірте тим самим запитом через Python

### Як зрозуміти, що саме не так — з'єднання чи модель?

- **Немає з'єднання**: зазвичай буде `ConnectionError` або таймаут. У логах програми з'явиться підказка українською про `ollama serve` і перевірку `http://localhost:11434`.
- **Модель не знайдено**: Ollama часто повертає `404` з текстом на кшталт `model '...' not found`. Програма тепер логує окреме зрозуміле повідомлення українською з порадою перевірити `ollama list` або виконати `ollama pull ...`.

### Які повідомлення тепер показує програма в логах?

Приклади:

- `Не вдалося з'єднатися з локальною Ollama. Запустіть ollama serve та повторіть спробу.`
- `Модель llava-custom не знайдено в Ollama. Звірте назву з ollama list або встановіть модель командою ollama pull llava-custom.`
- `Ollama відповіла 404. Перевірте URL, що працює саме сервер Ollama, і endpoint /api/generate.`

### Чи можна змінити модель без редагування коду?

Так. Можна:

- ввести модель у GUI-полі **Ollama модель**
- або задати змінну середовища `AUTOPHOTOSORTER_OLLAMA_MODEL`

### Чи потрібен `ollama run`?

Не обов'язково для роботи API, але це зручна ручна перевірка:

```bash
ollama run llava
```

Якщо `ollama run ...` не працює, спершу виправте це, а вже потім підключайте модель до AutoPhotoSorter.

---

## Вимоги до системи

- Python 3.10+
- tkinter (входить до стандартної бібліотеки Python; на Linux може вимагати `sudo apt-get install python3-tk`)
- Підтримувані формати зображень: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`, `.tif`
