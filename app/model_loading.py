import json
import os
import asyncio
import logging
from typing import Optional
import aiohttp
from settings import (
    COZE_API_BASE,
    COZE_API_TOKEN,
    COZE_BOT_ID,
    COZE_USER_ID,
    A302_API_KEY,
    A302_API_BASE,
    A302_MODEL_NAME,
)
from cozepy import AsyncTokenAuth, ChatStatus, Message, AsyncCoze, MessageType

logger = logging.getLogger(__name__)


class CozeLLM:
    """Async wrapper for Coze using AsyncCoze from cozepy."""

    def __init__(self, bot_id: str, user_id: str, api_token: str, api_base: str):
        self.bot_id = bot_id
        self.user_id = user_id
        self.coze = AsyncCoze(auth=AsyncTokenAuth(api_token), base_url=api_base)
        self.model_name = os.getenv("COZE_MODEL_NAME", "coze")

    async def ainvoke(self, prompt: str) -> str:
        """
        Submit prompt to Coze and return the RAW text content of the first
        ANSWER message.  JSON parsing is intentionally left to the caller
        (skills._parse_llm_json).
        """
        if isinstance(prompt, str):
            prompt = prompt.encode('utf-8').decode('utf-8')

        logger.info(f"[LLM] Creating chat with prompt size: {len(prompt)} chars")
        chat = await self.coze.chat.create(
            bot_id=self.bot_id,
            user_id=self.user_id,
            additional_messages=[Message.build_user_question_text(prompt)],
        )
        logger.info(f"[LLM] Chat created: {chat.id}, status: {chat.status}")

        # Poll until complete
        wait_count = 0
        while chat.status == ChatStatus.IN_PROGRESS:
            wait_count += 1
            if wait_count % 10 == 0:
                logger.info(f"[LLM] Still waiting... ({wait_count}s)")
            await asyncio.sleep(1)
            chat = await self.coze.chat.retrieve(
                conversation_id=chat.conversation_id,
                chat_id=chat.id
            )

        logger.info(f"[LLM] Chat completed with status: {chat.status} after {wait_count}s")

        try:
            messages = await self.coze.chat.messages.list(
                conversation_id=chat.conversation_id,
                chat_id=chat.id
            )
        except Exception as e:
            logger.error(f"Error retrieving messages: {e}")
            if hasattr(chat, 'last_error') and chat.last_error:
                return f"Error: {chat.last_error.msg}"
            return ""

        # Return the raw content of the first ANSWER message.
        # Do NOT extract the "response" key here — _parse_llm_json does that.
        for msg in messages:
            try:
                if hasattr(msg, 'type') and msg.type == MessageType.ANSWER:
                    raw = msg.content or ""
                    logger.info(f"[LLM] Raw answer (first 300 chars): {raw[:300]}")
                    return raw
            except Exception as e:
                logger.warning(f"Skipping message due to validation error: {e}")
                continue

        logger.warning("[LLM] No ANSWER message found in chat response.")
        return ""


def load_coze_llm() -> CozeLLM | None:
    """Convenience factory: returns a CozeLLM or None if env not configured."""
    bot_id = COZE_BOT_ID
    user_id = COZE_USER_ID
    token = COZE_API_TOKEN
    base_api = COZE_API_BASE
    if not (bot_id and user_id and token and base_api):
        return None
    return CozeLLM(bot_id=bot_id, user_id=user_id, api_token=token, api_base=base_api)


class A302LLM:
    """Simple async 302.ai client following official docs.

    POST /v1/chat/completions?async=true -> returns {"task_id": "..."}
    GET  /async_result?task_id=...       -> polls until done
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = A302_API_BASE,
        model_name: str = A302_MODEL_NAME,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def ainvoke(self, prompt: str) -> str:
        """
        Submit async chat request, poll for result, and return the RAW text
        content string from the model.  JSON parsing is left to the caller
        (skills._parse_llm_json).
        """
        session = await self._get_session()

        # Step 1: Submit async request
        logger.info(f"[302] Submitting async request ({len(prompt)} chars, model={self.model_name})")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }

        async with session.post(
            f"{self.base_url}/v1/chat/completions",
            params={"async": "true"},
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            result = await resp.json()
            task_id = result.get("task_id")
            if not task_id:
                raise ValueError(f"No task_id in response: {result}")
            logger.info(f"[302] Got task_id: {task_id}")

        # Step 2: Poll for result
        poll_url = f"{self.base_url}/async_result"
        start_time = asyncio.get_event_loop().time()
        poll_count = 0
        last_log_time = start_time

        while True:
            poll_count += 1
            await asyncio.sleep(1.0 if poll_count < 10 else 2.0)

            async with session.get(
                poll_url,
                params={"task_id": task_id},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                result = await resp.json()

                status_code = result.get("status_code")
                err = result.get("err", "")
                data = result.get("data")

                current_time = asyncio.get_event_loop().time()
                elapsed = current_time - start_time
                if current_time - last_log_time >= 10.0:
                    logger.info(
                        f"[302] Polling... {elapsed:.1f}s elapsed "
                        f"(poll #{poll_count}, status: {status_code}, err: {err})"
                    )
                    last_log_time = current_time

                if status_code == 200 and data and err not in ("result pending", "pending"):
                    logger.info(f"[302] Completed in {elapsed:.1f}s")
                    raw = self._extract_raw_text(data)
                    logger.info(f"[302] Raw text (first 300 chars): {raw[:300]}")
                    return raw

                if elapsed > 300:
                    logger.warning(f"[302] Timeout after {elapsed:.0f}s")
                    raise TimeoutError(f"Async result timeout. Last: {result}")

    def _extract_raw_text(self, data) -> str:
        """
        Extract the raw content string from the API response data object.

        This intentionally returns the model's text verbatim — it does NOT
        peek inside any JSON the model may have embedded.  That is the job
        of skills._parse_llm_json.

        Priority:
          1. data dict  → choices[0].message.content  (OpenAI-compatible)
          2. data dict  → direct "content" key
          3. data str   → parse as OpenAI JSON, grab content
          4. data str   → return as-is
          5. anything else → str(data)
        """
        # ── Case 1: dict with OpenAI-style choices ────────────────────────
        if isinstance(data, dict):
            if "choices" in data:
                try:
                    msg = data["choices"][0].get("message") or {}
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content
                except Exception as e:
                    logger.warning(f"[302] Could not read choices[0].message.content: {e}")

            # Direct "content" field (non-standard but seen in some providers)
            if "content" in data:
                return str(data["content"])

            # Last resort: dump the whole dict so _parse_llm_json can try
            logger.warning("[302] Unrecognised data dict shape; returning JSON dump.")
            return json.dumps(data, ensure_ascii=False)

        # ── Case 2: string data ───────────────────────────────────────────
        if isinstance(data, str):
            # It might itself be a serialised OpenAI response
            try:
                parsed = json.loads(data)
                if isinstance(parsed, dict) and "choices" in parsed:
                    msg = parsed["choices"][0].get("message") or {}
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content
            except (json.JSONDecodeError, TypeError, KeyError):
                pass
            # Otherwise treat the string as the raw model output directly
            return data

        # ── Case 3: unknown type ──────────────────────────────────────────
        logger.warning(f"[302] Unexpected data type {type(data)}; coercing to str.")
        return "" if data is None else str(data)

    async def aclose(self):
        if self._session and not self._session.closed:
            await self._session.close()


def load_302_llm() -> Optional[A302LLM]:
    """Factory: returns a configured 302.ai LLM or None if env is missing."""
    if not A302_API_KEY:
        return None
    return A302LLM(api_key=A302_API_KEY, base_url=A302_API_BASE, model_name=A302_MODEL_NAME)
