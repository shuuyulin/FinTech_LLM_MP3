import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
DB_PATH             = os.path.join(os.path.dirname(__file__), "..", "data", "stocks.db")

client = OpenAI(api_key=OPENAI_API_KEY)
