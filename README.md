# reflector

## Описание
Reflector — это Telegram-бот для проведения рефлексивных сессий по трёхфазной методике. Бот помогает пользователю структурировать анализ прошедшей недели, выявить ключевые эпизоды, провести рефлексию и спланировать новые действия.

## Основные возможности
- Трёхфазная рефлексия: анализ ситуации, рефлексивное действие, проектирование нового способа действия
- Интеграция с Telegram для удобного взаимодействия
- Поддержка нескольких языковых моделей (Coze, 302.ai)
- Импорт и обработка PDF-документов для расширения базы знаний (RAG)
- Векторный поиск по знаниям с помощью LangChain и ChromaDB
- Асинхронная работа с базой данных (aiosqlite)

## Структура проекта

```
reflector/
├── app/
│   ├── bot.py           # Основной интерфейс Telegram-бота, конечный автомат фаз
│   ├── skills.py        # Асинхронные функции-скиллы для каждой фазы/слоя
│   ├── model_loading.py # Загрузка и работа с LLM (Coze, 302.ai)
│   ├── rag.py           # Обработка PDF, создание векторных представлений
│   ├── settings.py      # Загрузка конфигурации и переменных окружения
├── logs/
│   └── usage_log.json   # Журнал использования бота
├── evaluation_set.csv   # Набор для тестирования/оценки
├── requirements.txt     # pip-зависимости
├── environment.yml      # conda-окружение
├── README.md            # Описание проекта
```

## Установка и запуск

1. Клонируйте репозиторий и создайте виртуальное окружение:
	```bash
	git clone <repo_url>
	cd reflector
	python -m venv venv
	source venv/bin/activate  # или venv\Scripts\activate для Windows
	```
2. Установите зависимости:
	```bash
	pip install -r requirements.txt
	```
	или через conda:
	```bash
	conda env create -f environment.yml
	conda activate hobsbawm-bot
	```
3. Создайте файл .env и заполните переменные (см. app/settings.py):
	- TELEGRAM_BOT_TOKEN
	- COZE_API_TOKEN, COZE_API_BASE, COZE_BOT_ID, COZE_USER_ID
	- API_302AI_KEY, A302_API_BASE, A302_MODEL_NAME
	- PRIMARY_DB_PATH, PRIMARY_DB_Documents_PATH
4. Запустите бота:
	```bash
	python app/bot.py
	```

## Используемые технологии
- Python 3.11+
- pyTelegramBotAPI
- LangChain, ChromaDB, HuggingFace Transformers
- PDFPlumber
- aiosqlite
- dotenv

## Лицензия
MIT