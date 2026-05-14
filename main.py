"""
main.py — GUI-застосунок AutoPhotoSorter.

Автоматичний аналіз, сортування та перейменування фотографій товарів.

Запуск:
  python main.py

Вимоги: дивіться requirements.txt
"""

import logging
import os
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import json

from analyzer import (
    DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_URL, OLLAMA_MAX_IMAGES_PER_RANK,
    get_ollama_models, _CLASSIFY_PROMPT, _RANK_PROMPT_TEMPLATE, _CLIP_TEXT_PROMPTS,
    set_custom_prompts
)
from sorter import find_subfolders_with_images, get_images_in_folder, rename_images_in_folder, process_folder
from reporter import generate_report

# ---------------------------------------------------------------------------
# Налаштування логування
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Константи інтерфейсу
# ---------------------------------------------------------------------------
APP_TITLE = "AutoPhotoSorter — Сортування фотографій товарів"
APP_VERSION = "1.0.0"
WINDOW_MIN_WIDTH = 860
WINDOW_MIN_HEIGHT = 700
DEFAULT_GEOMETRY = "960x750"
UI_INIT_DELAY_MS = 100  # Delay for UI initialization before binding clipboard events

# Шлях до конфігураційного файлу
CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".autophotosorter_config.json")

_API_TYPE_OPTIONS = [
    ("Тільки OpenCV (без AI)", "none"),
    ("Google Gemini API", "gemini"),
    ("OpenAI Vision API", "openai"),
    ("Ollama (локальна модель)", "ollama"),
    ("Локальна модель CLIP", "clip"),
]

_API_HINTS = {
    "none": "💡 Лише OpenCV: визначає білий фон. Без AI класифікації типу контенту.",
    "gemini": "💡 Потрібен Google Gemini API ключ: https://aistudio.google.com/app/apikey\nУвага: ключ буде видимий в полі вводу.",
    "openai": "💡 Потрібен OpenAI API ключ: https://platform.openai.com/api-keys\nУвага: ключ буде видимий в полі вводу.",
    "ollama": (
        f"💡 Ollama: локальний сервер з vision-моделями. URL за замовчуванням: {DEFAULT_OLLAMA_URL}\n"
        "Для фото використовуйте vision-модель (llava, bakllava) і назву точно з 'ollama list'.\n"
        f"✨ Новинка: Ollama аналізує всі фото товару разом (до {OLLAMA_MAX_IMAGES_PER_RANK} шт.) "
        "і сама визначає оптимальний порядок зображень у папці."
    ),
    "clip": (
        "💡 CLIP: локальна модель (~350 MB), завантажується при першому запуску. "
        "Потрібні: pip install torch torchvision clip"
    ),
}


# ---------------------------------------------------------------------------
# Головний клас застосунку
# ---------------------------------------------------------------------------

class AutoPhotoSorterApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry(DEFAULT_GEOMETRY)
        self.root.minsize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        # ---- Змінні стану ----
        self._input_folder = tk.StringVar()
        self._api_type = tk.StringVar(value="none")
        self._api_key = tk.StringVar()
        self._ollama_url = tk.StringVar(value=DEFAULT_OLLAMA_URL)
        self._ollama_model = tk.StringVar(value=DEFAULT_OLLAMA_MODEL)
        self._ollama_models_list = []  # Список доступних моделей
        self._dry_run = tk.BooleanVar(value=False)
        self._processing = False
        self._stop_requested = False
        
        # Промти для AI
        self._classify_prompt = tk.StringVar()
        self._rank_prompt = tk.StringVar()
        
        # Завантажити конфігурацію (встановить промти з файлу або за замовчуванням)
        self._load_config()

        self._build_ui()
        
        # Bind Ctrl+C/Ctrl+V для всіх Entry widgets
        self._setup_clipboard_bindings()

    # -----------------------------------------------------------------------
    # Конфігурація
    # -----------------------------------------------------------------------
    
    def _load_config(self) -> None:
        """Завантажує конфігурацію з файлу."""
        # Встановлюємо значення за замовчуванням
        self._classify_prompt.set(_CLASSIFY_PROMPT)
        self._rank_prompt.set(_RANK_PROMPT_TEMPLATE)
        
        if not os.path.exists(CONFIG_FILE):
            return
        
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if 'classify_prompt' in config:
                self._classify_prompt.set(config['classify_prompt'])
            if 'rank_prompt' in config:
                self._rank_prompt.set(config['rank_prompt'])
            
            logger.info("Конфігурацію завантажено з %s", CONFIG_FILE)
        except Exception as exc:
            logger.warning("Помилка завантаження конфігурації: %s", exc)
    
    def _save_config(self) -> None:
        """Зберігає конфігурацію у файл."""
        try:
            config = {
                'classify_prompt': self._classify_prompt.get(),
                'rank_prompt': self._rank_prompt.get(),
            }
            
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            logger.info("Конфігурацію збережено в %s", CONFIG_FILE)
            messagebox.showinfo("Успіх", "Налаштування промтів збережено!")
        except Exception as exc:
            logger.error("Помилка збереження конфігурації: %s", exc)
            messagebox.showerror("Помилка", f"Не вдалося зберегти налаштування:\n{exc}")
    
    def _setup_clipboard_bindings(self) -> None:
        """Налаштовує Ctrl+C/Ctrl+V для Entry widgets."""
        # Знаходимо всі Entry віджети і додаємо до них підтримку копіювання/вставки
        def bind_entry(widget):
            if isinstance(widget, ttk.Entry):
                widget.bind('<Control-c>', lambda e: self._copy_to_clipboard(widget))
                widget.bind('<Control-v>', lambda e: self._paste_from_clipboard(widget))
                widget.bind('<Control-x>', lambda e: self._cut_to_clipboard(widget))
                # Додаємо контекстне меню
                self._add_context_menu(widget)
            for child in widget.winfo_children():
                bind_entry(child)
        
        # Wait for UI initialization before binding
        self.root.after(UI_INIT_DELAY_MS, lambda: bind_entry(self.root))
    
    def _add_context_menu(self, entry_widget: ttk.Entry) -> None:
        """Додає контекстне меню з опціями копіювання/вставки до Entry widget."""
        menu = tk.Menu(entry_widget, tearoff=0)
        menu.add_command(label="Вирізати", command=lambda: self._cut_to_clipboard(entry_widget))
        menu.add_command(label="Копіювати", command=lambda: self._copy_to_clipboard(entry_widget))
        menu.add_command(label="Вставити", command=lambda: self._paste_from_clipboard(entry_widget))
        menu.add_separator()
        menu.add_command(label="Виділити все", command=lambda: entry_widget.select_range(0, tk.END))
        
        def show_menu(event):
            menu.tk_popup(event.x_root, event.y_root)
        
        entry_widget.bind('<Button-3>', show_menu)
    
    def _copy_to_clipboard(self, entry_widget: ttk.Entry) -> None:
        """Копіює виділений текст з Entry widget в буфер обміну."""
        try:
            if entry_widget.selection_present():
                text = entry_widget.selection_get()
                self.root.clipboard_clear()
                self.root.clipboard_append(text)
        except tk.TclError:
            pass
    
    def _cut_to_clipboard(self, entry_widget: ttk.Entry) -> None:
        """Вирізає виділений текст з Entry widget в буфер обміну."""
        try:
            if entry_widget.selection_present():
                text = entry_widget.selection_get()
                self.root.clipboard_clear()
                self.root.clipboard_append(text)
                # Видаляємо виділений текст
                start = entry_widget.index(tk.SEL_FIRST)
                end = entry_widget.index(tk.SEL_LAST)
                entry_widget.delete(start, end)
        except tk.TclError:
            pass
    
    def _paste_from_clipboard(self, entry_widget: ttk.Entry) -> None:
        """Вставляє текст з буфера обміну в Entry widget."""
        try:
            text = self.root.clipboard_get()
            if entry_widget.selection_present():
                # Заміняємо виділений текст
                start = entry_widget.index(tk.SEL_FIRST)
                end = entry_widget.index(tk.SEL_LAST)
                entry_widget.delete(start, end)
                entry_widget.insert(start, text)
            else:
                # Вставляємо в позицію курсора
                entry_widget.insert(tk.INSERT, text)
        except tk.TclError:
            pass

    # -----------------------------------------------------------------------
    # Побудова інтерфейсу
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        outer = ttk.Frame(self.root, padding=12)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(1, weight=1)

        row = 0

        # --- Заголовок ---
        ttk.Label(
            outer,
            text="🗂️  AutoPhotoSorter",
            font=("Helvetica", 18, "bold"),
        ).grid(row=row, column=0, columnspan=3, pady=(0, 2))
        row += 1

        ttk.Label(
            outer,
            text="Автоматичний аналіз, сортування та перейменування фотографій товарів",
            font=("Helvetica", 10),
            foreground="#555555",
        ).grid(row=row, column=0, columnspan=3, pady=(0, 10))
        row += 1

        ttk.Separator(outer, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=6
        )
        row += 1

        # --- Вхідна папка ---
        ttk.Label(outer, text="📁  Вхідна папка:").grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Entry(outer, textvariable=self._input_folder).grid(
            row=row, column=1, sticky="ew", pady=4
        )
        ttk.Button(outer, text="Огляд…", width=10,
                   command=self._browse_input).grid(
            row=row, column=2, padx=(6, 0), pady=4
        )
        row += 1

        ttk.Separator(outer, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=6
        )
        row += 1

        # --- Налаштування AI ---
        api_frame = ttk.LabelFrame(outer, text="  🤖  Метод аналізу  ", padding=8)
        api_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        api_frame.columnconfigure(1, weight=1)
        row += 1

        radio_row_frame = ttk.Frame(api_frame)
        radio_row_frame.grid(row=0, column=0, columnspan=2, sticky="w")

        for i, (label, value) in enumerate(_API_TYPE_OPTIONS):
            ttk.Radiobutton(
                radio_row_frame,
                text=label,
                variable=self._api_type,
                value=value,
                command=self._on_api_type_change,
            ).grid(row=0, column=i, padx=(0, 14), sticky="w")

        ttk.Label(api_frame, text="API ключ:").grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )
        self._api_key_entry = ttk.Entry(
            api_frame, textvariable=self._api_key, state="disabled"
        )
        self._api_key_entry.grid(
            row=1, column=1, sticky="ew", pady=(6, 0), padx=(6, 0)
        )

        # --- Ollama fields ---
        ttk.Label(api_frame, text="Ollama URL:").grid(
            row=2, column=0, sticky="w", pady=(6, 0)
        )
        self._ollama_url_entry = ttk.Entry(
            api_frame, textvariable=self._ollama_url, state="disabled"
        )
        self._ollama_url_entry.grid(
            row=2, column=1, sticky="ew", pady=(6, 0), padx=(6, 0)
        )

        # Ollama модель - Combobox замість Entry
        ollama_model_frame = ttk.Frame(api_frame)
        ollama_model_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ollama_model_frame.columnconfigure(1, weight=1)
        
        ttk.Label(ollama_model_frame, text="Ollama модель:").grid(
            row=0, column=0, sticky="w"
        )
        self._ollama_model_combo = ttk.Combobox(
            ollama_model_frame, textvariable=self._ollama_model, state="disabled"
        )
        self._ollama_model_combo.grid(
            row=0, column=1, sticky="ew", padx=(6, 6)
        )
        self._ollama_refresh_btn = ttk.Button(
            ollama_model_frame, text="🔄 Оновити список", width=18,
            command=self._refresh_ollama_models, state="disabled"
        )
        self._ollama_refresh_btn.grid(
            row=0, column=2, padx=(0, 0)
        )

        self._api_hint_label = ttk.Label(
            api_frame,
            text=_API_HINTS["none"],
            foreground="#666666",
            font=("Helvetica", 9),
            wraplength=700,
        )
        self._api_hint_label.grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

        # --- Налаштування промтів AI ---
        prompts_frame = ttk.LabelFrame(outer, text="  ✏️  Налаштування промтів AI  ", padding=8)
        prompts_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        prompts_frame.columnconfigure(0, weight=1)
        row += 1
        
        # Кнопка для відкриття редактора промтів
        ttk.Button(
            prompts_frame, 
            text="📝 Редагувати промти для класифікації та ранжування",
            command=self._open_prompt_editor
        ).grid(row=0, column=0, sticky="ew", pady=2)

        # --- Опції ---
        opts_frame = ttk.Frame(outer)
        opts_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 8))
        row += 1

        ttk.Checkbutton(
            opts_frame,
            text="🔍  Тестовий режим — аналізувати без перейменування файлів",
            variable=self._dry_run,
        ).grid(row=0, column=0, sticky="w")

        # --- Кнопки дій ---
        btn_frame = ttk.Frame(outer)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=(0, 10))
        row += 1

        self._start_btn = ttk.Button(
            btn_frame, text="▶  Почати обробку",
            command=self._start_processing, width=20
        )
        self._start_btn.grid(row=0, column=0, padx=5)

        self._stop_btn = ttk.Button(
            btn_frame, text="⏹  Зупинити",
            command=self._request_stop, state="disabled", width=14
        )
        self._stop_btn.grid(row=0, column=1, padx=5)

        ttk.Button(
            btn_frame, text="🗑  Очистити лог",
            command=self._clear_log, width=14
        ).grid(row=0, column=2, padx=5)

        # --- Прогрес ---
        ttk.Label(outer, text="Прогрес:").grid(
            row=row, column=0, sticky="w", pady=(0, 2)
        )
        row += 1

        self._progress_var = tk.DoubleVar(value=0.0)
        self._progress_bar = ttk.Progressbar(
            outer, variable=self._progress_var, maximum=100
        )
        self._progress_bar.grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=2
        )
        row += 1

        self._progress_label = ttk.Label(
            outer, text="Готово до роботи", foreground="#777777"
        )
        self._progress_label.grid(
            row=row, column=0, columnspan=3, sticky="w"
        )
        row += 1

        # --- Журнал ---
        ttk.Label(outer, text="📋  Журнал роботи:").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(10, 2)
        )
        row += 1

        outer.rowconfigure(row, weight=1)
        self._log_box = scrolledtext.ScrolledText(
            outer, height=14, wrap=tk.WORD,
            font=("Courier", 9), state="disabled",
            background="#1e1e1e", foreground="#d4d4d4",
        )
        self._log_box.grid(
            row=row, column=0, columnspan=3, sticky="nsew", pady=2
        )

        # Теги кольорів для лога
        self._log_box.tag_configure("info",    foreground="#d4d4d4")
        self._log_box.tag_configure("success", foreground="#6dd672")
        self._log_box.tag_configure("warning", foreground="#e5c07b")
        self._log_box.tag_configure("error",   foreground="#e06c75")
        self._log_box.tag_configure("dim",     foreground="#888888")

    # -----------------------------------------------------------------------
    # Обробники подій UI
    # -----------------------------------------------------------------------

    def _on_api_type_change(self) -> None:
        api = self._api_type.get()
        needs_key = api in ("gemini", "openai")
        needs_ollama = api == "ollama"
        
        self._api_key_entry.config(state="normal" if needs_key else "disabled")
        self._ollama_url_entry.config(state="normal" if needs_ollama else "disabled")
        self._ollama_model_combo.config(state="readonly" if needs_ollama else "disabled")
        self._ollama_refresh_btn.config(state="normal" if needs_ollama else "disabled")
        self._api_hint_label.config(text=_API_HINTS.get(api, ""))

    def _browse_input(self) -> None:
        folder = filedialog.askdirectory(
            title="Виберіть папку з підпапками фотографій товарів"
        )
        if folder:
            self._input_folder.set(folder)
    
    def _refresh_ollama_models(self) -> None:
        """Оновлює список доступних моделей Ollama."""
        ollama_url = self._ollama_url.get().strip()
        if not ollama_url:
            messagebox.showerror("Помилка", "Будь ласка, вкажіть Ollama URL.")
            return
        
        self._log("🔄  Отримання списку моделей Ollama...", "info")
        
        # Викликаємо в окремому потоці, щоб не блокувати GUI
        def fetch_models():
            models = get_ollama_models(ollama_url)
            self.root.after(0, lambda: self._update_ollama_models(models))
        
        threading.Thread(target=fetch_models, daemon=True).start()
    
    def _update_ollama_models(self, models: list[str]) -> None:
        """Оновлює випадаючий список моделей Ollama."""
        if not models:
            self._log("   ⚠️  Не вдалося отримати список моделей", "warning")
            messagebox.showwarning(
                "Попередження",
                "Не вдалося отримати список моделей з Ollama.\n"
                "Перевірте, що Ollama запущено і URL правильний."
            )
            return
        
        self._ollama_models_list = models
        self._ollama_model_combo['values'] = models
        
        # Якщо поточна модель не в списку, встановлюємо першу
        current_model = self._ollama_model.get()
        if current_model not in models and models:
            self._ollama_model.set(models[0])
        
        self._log(f"   ✅  Знайдено моделей: {len(models)}", "success")
        messagebox.showinfo("Успіх", f"Знайдено {len(models)} моделей Ollama.")
    
    def _open_prompt_editor(self) -> None:
        """Відкриває вікно редагування промтів."""
        editor = tk.Toplevel(self.root)
        editor.title("Редагування промтів AI")
        editor.geometry("800x600")
        editor.transient(self.root)
        
        # Фрейм для промтів
        frame = ttk.Frame(editor, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        frame.rowconfigure(3, weight=1)
        
        # Промт класифікації
        ttk.Label(frame, text="Промт для класифікації зображень:", font=("Helvetica", 10, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 4)
        )
        
        classify_text = scrolledtext.ScrolledText(frame, height=10, wrap=tk.WORD, font=("Courier", 9))
        classify_text.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        classify_text.insert("1.0", self._classify_prompt.get())
        
        # Промт ранжування
        ttk.Label(frame, text="Промт для ранжування зображень:", font=("Helvetica", 10, "bold")).grid(
            row=2, column=0, sticky="w", pady=(0, 4)
        )
        
        rank_text = scrolledtext.ScrolledText(frame, height=10, wrap=tk.WORD, font=("Courier", 9))
        rank_text.grid(row=3, column=0, sticky="nsew", pady=(0, 10))
        rank_text.insert("1.0", self._rank_prompt.get())
        
        # Кнопки
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=4, column=0, pady=(10, 0))
        
        def save_prompts():
            self._classify_prompt.set(classify_text.get("1.0", tk.END).strip())
            self._rank_prompt.set(rank_text.get("1.0", tk.END).strip())
            self._save_config()
            editor.destroy()
        
        def reset_prompts():
            if messagebox.askyesno("Підтвердження", "Скинути промти до значень за замовчуванням?"):
                classify_text.delete("1.0", tk.END)
                classify_text.insert("1.0", _CLASSIFY_PROMPT)
                rank_text.delete("1.0", tk.END)
                rank_text.insert("1.0", _RANK_PROMPT_TEMPLATE)
        
        ttk.Button(btn_frame, text="💾 Зберегти", command=save_prompts, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="🔄 Скинути", command=reset_prompts, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="❌ Скасувати", command=editor.destroy, width=15).pack(side=tk.LEFT, padx=5)

    def _clear_log(self) -> None:
        self._log_box.config(state="normal")
        self._log_box.delete("1.0", tk.END)
        self._log_box.config(state="disabled")

    # -----------------------------------------------------------------------
    # Логування в UI
    # -----------------------------------------------------------------------

    def _log(self, message: str, tag: str = "info") -> None:
        """Потокобезпечне додавання рядка до журналу."""
        def _append():
            self._log_box.config(state="normal")
            self._log_box.insert(tk.END, message + "\n", tag)
            self._log_box.see(tk.END)
            self._log_box.config(state="disabled")

        self.root.after(0, _append)

    def _set_progress(self, value: float, text: str = "") -> None:
        def _update():
            self._progress_var.set(value)
            if text:
                self._progress_label.config(text=text)
        self.root.after(0, _update)

    # -----------------------------------------------------------------------
    # Запуск / зупинка обробки
    # -----------------------------------------------------------------------

    def _start_processing(self) -> None:
        input_dir = self._input_folder.get().strip()

        if not input_dir:
            messagebox.showerror("Помилка", "Будь ласка, вкажіть вхідну папку.")
            return
        if not os.path.isdir(input_dir):
            messagebox.showerror("Помилка", f"Папка не існує:\n{input_dir}")
            return

        # Генеруємо шлях до звіту автоматично в папці з фото
        output_report = os.path.join(input_dir, "report.xlsx")

        # Встановлюємо користувацькі промти (завжди встановлюємо поточні значення)
        classify_prompt = self._classify_prompt.get().strip()
        rank_prompt = self._rank_prompt.get().strip()
        
        # Передаємо None якщо промт відповідає стандартному (для економії пам'яті)
        set_custom_prompts(
            None if classify_prompt == _CLASSIFY_PROMPT else classify_prompt,
            None if rank_prompt == _RANK_PROMPT_TEMPLATE else rank_prompt
        )

        api = self._api_type.get()
        api_key = self._api_key.get().strip()
        ollama_url = self._ollama_url.get().strip()
        ollama_model = self._ollama_model.get().strip()
        
        if api in ("gemini", "openai") and not api_key:
            if not messagebox.askyesno(
                "API ключ відсутній",
                f"API ключ для {api.upper()} не вказано.\n"
                "Буде використано лише OpenCV.\nПродовжити?",
            ):
                return
            api = "none"
        
        if api == "ollama" and not ollama_url:
            messagebox.showerror("Помилка", "Будь ласка, вкажіть Ollama URL.")
            return

        self._processing = True
        self._stop_requested = False
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._set_progress(0.0, "Ініціалізація…")

        thread = threading.Thread(
            target=self._worker,
            args=(input_dir, output_report, api, api_key, ollama_url, ollama_model, self._dry_run.get()),
            daemon=True,
        )
        thread.start()

    def _request_stop(self) -> None:
        self._stop_requested = True
        self._log("⏹  Запит на зупинку отримано…", "warning")

    def _on_processing_finished(self) -> None:
        self._processing = False
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")

    # -----------------------------------------------------------------------
    # Робочий потік
    # -----------------------------------------------------------------------

    def _worker(self,
                input_dir: str,
                output_report: str,
                api_type: str,
                api_key: str,
                ollama_url: str,
                ollama_model: str,
                dry_run: bool) -> None:
        try:
            self._run_processing(input_dir, output_report, api_type, api_key, 
                               ollama_url, ollama_model, dry_run)
        except Exception as exc:
            self._log(f"❌  Критична помилка: {exc}", "error")
            logger.exception("Worker thread error")
        finally:
            self.root.after(0, self._on_processing_finished)

    def _run_processing(self,
                        input_dir: str,
                        output_report: str,
                        api_type: str,
                        api_key: str,
                        ollama_url: str,
                        ollama_model: str,
                        dry_run: bool) -> None:

        self._log(f"🚀  Початок обробки: {input_dir}", "info")
        self._log(f"   Метод аналізу: {api_type}", "dim")
        if dry_run:
            self._log("   ⚠️  Тестовий режим — файли НЕ будуть перейменовані", "warning")

        # --- Знаходимо всі папки з зображеннями ---
        folders = find_subfolders_with_images(input_dir)

        if not folders:
            self._log("⚠️  Не знайдено папок із зображеннями.", "warning")
            self._set_progress(100.0, "Завершено — нічого не знайдено")
            return

        self._log(f"📂  Знайдено папок для обробки: {len(folders)}", "info")

        all_results: list[dict] = []

        for folder_idx, folder_path in enumerate(folders):
            if self._stop_requested:
                self._log("⏹  Обробку зупинено.", "warning")
                break

            folder_name = os.path.basename(folder_path)
            pct = folder_idx / len(folders) * 100
            self._set_progress(pct, f"Обробка: {folder_name}")
            self._log(f"\n📁  {folder_name}", "info")

            # Callback для прогресу аналізу файлів
            def _progress_cb(i: int, total: int, msg: str) -> None:
                if not self._stop_requested:
                    self._log(f"   {msg}", "dim")

            result = process_folder(
                folder_path,
                api_type=api_type,
                api_key=api_key or None,
                ollama_url=ollama_url or None,
                ollama_model=ollama_model or DEFAULT_OLLAMA_MODEL,
                progress_callback=_progress_cb,
            )

            # --- Відображення результатів папки ---
            if result.get("error") == "No images found":
                self._log("   ⚠️  Немає зображень у папці", "warning")
            elif result.get("error"):
                self._log(f"   ❌  Помилка: {result['error']}", "error")
            else:
                count = len(result.get("sorted_images", []))
                self._log(f"   ✅  Знайдено зображень: {count}", "success")

                if result.get("has_ideal_main"):
                    self._log("   ✅  Ідеальне головне фото знайдено", "success")
                elif result.get("fallback_used"):
                    fb = result.get("fallback_image") or {}
                    self._log(
                        f"   ⚠️  Головне фото не знайдено. Альтернатива: "
                        f"{fb.get('filename', 'N/A')}",
                        "warning",
                    )

                if result.get("ollama_ranked"):
                    self._log(
                        "   ✅  Порядок фото визначено Ollama (ранжування набору)",
                        "success",
                    )

                # --- Перейменування ---
                if result.get("sorted_images"):
                    try:
                        renames = rename_images_in_folder(
                            folder_path,
                            result["sorted_images"],
                            dry_run=dry_run,
                        )
                        result["renamed_files"] = renames
                        action = "Буде перейменовано" if dry_run else "Перейменовано"
                        self._log(f"   ✅  {action} файлів: {len(renames)}", "success")
                        for r in renames:
                            self._log(
                                f"      {r['old_name']}  →  {r['new_name']}"
                                f"  [{r['category']}]",
                                "dim",
                            )
                    except Exception as exc:
                        self._log(f"   ❌  Помилка перейменування: {exc}", "error")

            all_results.append(result)

        # --- Генерація звіту ---
        if all_results:
            try:
                report_path = generate_report(all_results, output_report)
                self._log(f"\n📊  Звіт збережено: {report_path}", "success")
                self.root.after(0, self._ask_open_report, report_path)
            except Exception as exc:
                self._log(f"\n❌  Помилка генерації звіту: {exc}", "error")

        # --- Підсумок ---
        total = len(all_results)
        ok = sum(1 for r in all_results if not r.get("error"))
        warn = sum(1 for r in all_results if r.get("fallback_used"))
        err = sum(1 for r in all_results if r.get("error"))

        self._log("\n" + "─" * 55, "dim")
        self._log(f"✅  Оброблено папок: {total}", "info")
        self._log(
            f"   Успішно: {ok}  |  Попередження: {warn}  |  Помилки: {err}",
            "info",
        )
        self._set_progress(100.0, "Завершено!")

    # -----------------------------------------------------------------------
    # Відкриття звіту
    # -----------------------------------------------------------------------

    def _ask_open_report(self, report_path: str) -> None:
        if messagebox.askyesno(
            "Готово!",
            f"Обробку завершено!\n\nВідкрити звіт?\n{report_path}",
        ):
            _open_file(report_path)


# ---------------------------------------------------------------------------
# Утиліта відкриття файлу
# ---------------------------------------------------------------------------

def _open_file(path: str) -> None:
    """Відкриває файл у системній програмі за замовчуванням."""
    try:
        if hasattr(os, "startfile"):  # Windows only
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform.startswith("darwin"):
            subprocess.call(["open", path])
        else:
            subprocess.call(["xdg-open", path])
    except Exception as exc:
        logger.warning("Не вдалося відкрити файл %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Точка входу
# ---------------------------------------------------------------------------

def main() -> None:
    root = tk.Tk()
    try:
        # Спроба встановити іконку (ігнорується якщо файл відсутній)
        root.iconbitmap(os.path.join(os.path.dirname(__file__), "icon.ico"))
    except Exception:
        pass
    app = AutoPhotoSorterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
