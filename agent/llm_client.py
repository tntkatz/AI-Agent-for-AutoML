import os
import time
import asyncio
import logging
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai import AsyncOpenAI, APIConnectionError, RateLimitError, APIStatusError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenClient:
    def __init__(self):
        self.client = AsyncOpenAI(timeout=15.0) 
        self.model_name = os.getenv("OPENAI_MODEL", "qwen/qwen3.8-27b")
        
        self.api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://changellenge.up.railway.app/v1")
        self.info_url = base_url.replace("/v1", "/key/info")
        
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost_rub = 0.0
        
        self.budget_limit_rub = 600.0
        self.cost_warning_threshold = 580.0
        
        self.start_time = time.time()
        self.max_session_time_sec = 2 * 3600 + 40 * 60

    async def fetch_current_spend(self) -> float:
        """Стучится к API организаторов за точным балансом."""
        if not self.api_key:
            return self.total_cost_rub
        try:
            async with httpx.AsyncClient() as http_client:
                resp = await http_client.get(
                    self.info_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=5.0
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("info", {}).get("spend", self.total_cost_rub)
        except Exception as e:
            logger.warning(f"Не удалось получить точный баланс: {e}")
        return self.total_cost_rub

    async def check_limits(self) -> bool:
        """Проверяет время и запрашивает актуальный баланс."""
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_session_time_sec:
            logger.warning("ТАЙМАУТ: Прошло 2 часа 40 минут. Мягкая остановка запросов.")
            return False
            
        self.total_cost_rub = await self.fetch_current_spend()
        
        if self.total_cost_rub >= self.cost_warning_threshold:
            logger.warning(f"БЮДЖЕТ: Потрачено {self.total_cost_rub:.2f} руб. (Близко к лимиту). Мягкая остановка.")
            return False
            
        return True

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((APIConnectionError, RateLimitError, APIStatusError)),
        before_sleep=lambda retry_state: logger.warning(f"Сбой сети, повторяем... (Попытка {retry_state.attempt_number})")
    )
    async def ask_llm(self, prompt: str) -> str:
        if not await self.check_limits():
            return "SYSTEM_STOP: Достигнут лимит времени или бюджета. Завершайте работу."

        logger.info(f"Отправка запроса к {self.model_name}...")
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
        except APIStatusError as e:
            if "Budget has been exceeded" in str(e.message):
                logger.error("Лимит в 600 рублей исчерпан на стороне сервера!")
                return "SYSTEM_STOP: Budget exceeded on server."
            if e.status_code >= 500:
                logger.warning(f"Ошибка сервера {e.status_code}. Уходим в retry...")
                raise e
            raise e
        
        p_tokens = response.usage.prompt_tokens
        c_tokens = response.usage.completion_tokens
        self.prompt_tokens += p_tokens
        self.completion_tokens += c_tokens
        
        self.total_cost_rub = await self.fetch_current_spend()
        logger.info(f"Токены: {p_tokens} (prompt) + {c_tokens} (completion). Точный расход: {self.total_cost_rub:.2f} руб.")
        
        return response.choices[0].message.content

async def run_test():
    client = QwenClient()
    print("\n--- Тестовый запрос ---")
    answer = await client.ask_llm("Назови столицу Франции одним словом.")
    print(f"Ответ ИИ: {answer}")
    
    print("\n--- Статистика модуля ---")
    print(f"Prompt токены: {client.prompt_tokens}")
    print(f"Completion токены: {client.completion_tokens}")
    print(f"Потрачено: {client.total_cost_rub:.4f} руб.")

if __name__ == "__main__":
    asyncio.run(run_test())