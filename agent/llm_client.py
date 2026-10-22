import os
import time
import asyncio
import logging
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception
from openai import AsyncOpenAI, APIConnectionError, RateLimitError, APIStatusError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_retryable(e):
    if isinstance(e, (APIConnectionError, RateLimitError)):
        return True
    if isinstance(e, APIStatusError) and e.status_code >= 500:
        return True
    return False

class QwenClient:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://changellenge.up.railway.app/v1")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=90.0
        ) 
        self.model_name = os.getenv("OPENAI_MODEL", "qwen/qwen3.8-27b")
        self.info_url = self.base_url.replace("/v1", "/key/info")
        
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost_rub = 0.0
        
        self.budget_limit_rub = 600.0
        self.cost_warning_threshold = 580.0
        
        self.start_time = time.time()
        self.max_session_time_sec = 2 * 3600 + 40 * 60

    async def fetch_current_spend(self) -> float:
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
            logger.warning(f"Balance fetch failed: {e}")
        return self.total_cost_rub

    async def check_limits(self) -> bool:
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_session_time_sec:
            logger.warning("Session timeout (2h 40m).")
            return False
            
        self.total_cost_rub = await self.fetch_current_spend()
        
        if self.total_cost_rub >= self.cost_warning_threshold:
            logger.warning(f"Budget warning: {self.total_cost_rub:.2f} RUB spent.")
            return False
            
        return True

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
        retry=retry_if_exception(is_retryable),
        before_sleep=lambda rs: logger.warning(f"Retry attempt {rs.attempt_number}/5")
    )
    async def ask_llm(self, prompt: str, system_prompt: str = None) -> str:
        if not await self.check_limits():
            return "SYSTEM_STOP"

        logger.info(f"Requesting {self.model_name}")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7
            )
        except APIStatusError as e:
            if "Budget has been exceeded" in str(e.message):
                logger.error("Budget limit exceeded (Server).")
                return "SYSTEM_STOP"
            raise e
        
        p_tokens = response.usage.prompt_tokens
        c_tokens = response.usage.completion_tokens
        self.prompt_tokens += p_tokens
        self.completion_tokens += c_tokens
        
        self.total_cost_rub = await self.fetch_current_spend()
        logger.info(f"Usage: {p_tokens+c_tokens} tokens. Total spend: {self.total_cost_rub:.2f} RUB.")
        
        return response.choices[0].message.content

async def run_test():
    client = QwenClient()
    print("\n--- Test Request ---")
    answer = await client.ask_llm(
        prompt="Назови столицу Франции.",
        system_prompt="Отвечай одним словом без пунктуации."
    )
    print(f"Response: {answer}")
    
    print("\n--- Stats ---")
    print(f"Prompt: {client.prompt_tokens}")
    print(f"Completion: {client.completion_tokens}")
    print(f"Spent: {client.total_cost_rub:.4f} RUB")

if __name__ == "__main__":
    asyncio.run(run_test())