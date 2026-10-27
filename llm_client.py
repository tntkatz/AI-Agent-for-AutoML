import os
import re
import time
import asyncio
import logging
from typing import Dict, List, Optional
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception
from openai import AsyncOpenAI, APIConnectionError, RateLimitError, APIStatusError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_retryable(e: Exception) -> bool:
    if isinstance(e, (APIConnectionError, RateLimitError, httpx.TimeoutException)):
        return True
    if isinstance(e, APIStatusError) and e.status_code >= 500:
        return True
    return False


class QwenClient:
    def __init__(self, start_time: Optional[float] = None):
        raw_key = os.getenv("OPENAI_API_KEY", "dummy-key")
        self.api_key = raw_key.strip().strip("'\"") if raw_key else "dummy-key"

        raw_url = os.getenv("OPENAI_BASE_URL", "https://changellenge.up.railway.app/v1")
        self.base_url = raw_url.strip().strip("'\"")

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=120.0,
            max_retries=0,
        )
        self.model_name = os.getenv("OPENAI_MODEL", "qwen/qwen3.8-27b").strip().strip("'\"")
        self.info_url = re.sub(r"/v1/?$", "/key/info", self.base_url)

        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost_rub = 0.0

        self.budget_limit_rub = 600.0
        self.cost_warning_threshold = 592.0

        self.start_time = start_time or time.time()
        self.max_session_time_sec = 2 * 3600 + 35 * 60

    def remaining_time_sec(self) -> float:
        return max(0.0, self.max_session_time_sec - (time.time() - self.start_time))

    async def fetch_current_spend(self) -> float:
        if not self.api_key or self.api_key == "dummy-key":
            return self.total_cost_rub
        try:
            async with httpx.AsyncClient(timeout=5.0) as http_client:
                resp = await http_client.get(
                    self.info_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, dict):
                        if "spend" in data and data["spend"] is not None:
                            return float(data["spend"])
                        info = data.get("info")
                        if isinstance(info, dict) and "spend" in info and info["spend"] is not None:
                            return float(info["spend"])
        except Exception as e:
            logger.debug(f"Balance fetch warning: {e}")
        return self.total_cost_rub

    async def check_limits(self) -> bool:
        if self.remaining_time_sec() <= 0:
            logger.warning("LLM session time limit reached (2h 35m). Proceeding to final ensemble.")
            return False

        self.total_cost_rub = await self.fetch_current_spend()

        if self.total_cost_rub >= self.cost_warning_threshold:
            logger.warning(f"Budget threshold approached: {self.total_cost_rub:.2f} RUB spent.")
            return False

        return True

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=12),
        stop=stop_after_attempt(5),
        retry=retry_if_exception(is_retryable),
        before_sleep=lambda rs: logger.warning(f"Retry attempt {rs.attempt_number}/5"),
    )
    async def ask_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.35,
    ) -> str:
        if not await self.check_limits():
            return "SYSTEM_STOP"

        logger.info(f"Requesting model: {self.model_name}")

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        try:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=8192,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
            except APIStatusError as inner_e:
                if inner_e.status_code in (400, 422):
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=8192,
                    )
                else:
                    raise inner_e
        except APIStatusError as e:
            err_text = str(getattr(e, "message", "")) + str(e)
            if "Budget has been exceeded" in err_text or e.status_code == 402:
                logger.error("Budget limit exceeded on proxy.")
                return "SYSTEM_STOP"
            raise e

        if response.usage:
            p_tokens = response.usage.prompt_tokens or 0
            c_tokens = response.usage.completion_tokens or 0
            self.prompt_tokens += p_tokens
            self.completion_tokens += c_tokens
            logger.info(f"Usage: {p_tokens + c_tokens} tokens (P={p_tokens}, C={c_tokens}).")

        self.total_cost_rub = await self.fetch_current_spend()
        logger.info(f"Current total spend: {self.total_cost_rub:.2f} RUB.")

        msg = response.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or ""
        if not content.strip() and reasoning:
            content = reasoning
        elif "def add_features" not in content and "def add_features" in reasoning:
            content = reasoning + "\n" + content

        return content