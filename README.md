# Alfa Hack: AI-Agent for AutoML

Аавтономный AI-агент для автоматического решения ML-задач (AutoML / AutoResearch)

---

## 1. Быстрый старт (Локальное окружение)

### Требования
* Python 3.12
* Git

### Установка зависимостей
```bash
# 1. Клонирование репозитория
git clone https://github.com/tntkatz/AI-Agent-for-AutoML
cd AI-Agent-for-AutoML

# 2. Создание и активация виртуального окружения
python3 -m venv .venv
source .venv/bin/activate

# 3. Установка зафиксированных пакетов
pip install -r requirements.txt
```

### Размещение датасетов
Папка `data/` добавлена в `.gitignore` и не отслеживается гитом. Скачанные папки с данными положите в локальную директорию `data/`:
```text
data/
├── regression/
│   ├── train.csv
│   ├── test.csv
│   ├── data_dictionary.csv
│   ├── sample_submission.csv
│   └── task.md
└── classification/
    ├── train.csv
    ├── test.csv
    ├── data_dictionary.csv
    ├── sample_submission.csv
    └── task.md
```

### Запуск текущего бейзлайна
Скрипт принимает аргументом путь к файлу описания задачи `task.md`:
```bash
# Тест регрессии (энергопотребление)
python solution.py data/regression/task.md

# Тест классификации (кибератаки)
python solution.py data/classification/task.md
```
После завершения работы в каталоге соответствующей задачи сформируется валидный файл `submission.csv`.

---

## 2. Структура проекта

```text
.
├── solution.py            # Единая точка входа для платформы проверки
├── baseline.py            # Исходный референс FastMCP от организаторов
├── requirements.txt       # Список зависимостей среды Docker
├── agent/                 # Пакет с модулями агента
│   ├── __init__.py
│   └── baseline_model.py  # Детерминированный Safety Baseline (LightGBM)
├── scripts/               # Вспомогательные скрипты упаковки и тестов
└── data/                  # Локальные датасеты (в Git НЕ коммитить!)
```

---

## 3. Git Workflow для команды (ПРАВИЛА РАБОТЫ)

Чтобы не ломать рабочий код и не разгребать бесконечные merge-конфликты, соблюдаем единый регламент:

### Главные запреты
1. **Категорически запрещено пушить напрямую в `main`!** Ветка `main` всегда должна оставаться стабильной и запускаемой.
2. **Никогда не добавлять в Git файлы `.csv`, архивы `.zip` и API-ключи!** Всегда проверяйте `git status` перед коммитом.

### Пошаговый цикл работы над задачей

1. **Синхронизация с `main` перед стартом:**
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Создание новой ветки под свою задачу:**
   Используем префиксы:
   * `feat/короткое-название` — новая функциональность
   * `fix/короткое-название` — исправление бага
   ```bash
   git checkout -b feat/task-name
   ```

3. **Фиксация изменений:**
   Добавляйте только те файлы, которые реально меняли (не используйте слепой `git add .`):
   ```bash
   git status
   git add agent/baseline_model.py
   git commit -m "feat: add cross-validation module"
   git push -u origin feat/task-name
   ```

4. **Pull Request (PR):**
   * Зайдите на GitHub в свой репозиторий.
   * Нажмите кнопку **Compare & pull request**.
   * Опишите, что было сделано и проверено.
   * Слияние в `main` выполняется только после просмотра кода тиммейтом.
   * После слияния удаляйте ветку на GitHub.

---

## 4. Распределение задач (Board Issues)

* **DE 1:** Клиент к API модели Qwen3.8-27B, учет лимитов токенов/бюджета (600 руб.), экспоненциальный retry при сбоях сети.
* **DE 2:** Изолированный runner выполнения кода (`subprocess`) с перехватом трейсбеков, таймер отсечки (2.5 часа), валидатор схемы `submission.csv`.
* **DS 1:** Локальный модуль кросс-валидации (5-fold KFold / StratifiedKFold) с точным расчетом целевых метрик (RMSE, Macro F1).
* **DS 2:** Генератор компактного профиля данных (EDA) из `data_dictionary.csv`, системные промпты для цикла AutoResearch (генерация гипотез и признаков).
