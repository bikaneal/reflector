import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

COZE_API_TOKEN = os.getenv("COZE_API_TOKEN")
COZE_API_BASE = os.getenv("COZE_API_BASE", "https://www.coze.com/")
COZE_BOT_ID = os.getenv("COZE_BOT_ID")
COZE_USER_ID = os.getenv("COZE_USER_ID")

PRIMARY_DB_PATH = os.path.join(PROJECT_ROOT, os.getenv("PRIMARY_DB_PATH"))

PRIMARY_DB_Documents_PATH = os.path.join(PROJECT_ROOT, os.getenv("PRIMARY_DB_Documents_PATH"))

# 302.ai configuration
# Accept common env var names, prefer API_302AI_KEY
A302_API_KEY = os.getenv("API_302AI_KEY")
A302_API_BASE = os.getenv("A302_API_BASE", "https://api.302.ai")
A302_MODEL_NAME = os.getenv("A302_MODEL_NAME", "gpt-4o")

# Provider switch: "coze" (default) or "302"
LLM_PROVIDER = (os.getenv("LLM_PROVIDER", "coze") or "coze").lower()