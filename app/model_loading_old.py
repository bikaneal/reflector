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
    """Async wrapper for Coze using AsyncCoze from cozepy.

    Uses the cozepy AsyncCoze client. Requires that `cozepy` is installed
    and `HAVE_COZE` is True.
    """

    def __init__(self, bot_id: str, user_id: str, api_token: str, api_base: str):
        self.bot_id = bot_id
        self.user_id = user_id
        self.coze = AsyncCoze(auth=AsyncTokenAuth(api_token), base_url=api_base)
        self.model_name = os.getenv("COZE_MODEL_NAME", "coze")
    
    async def ainvoke(self, prompt: str):
        if isinstance(prompt, str):
            prompt = prompt.encode('utf-8').decode('utf-8')

        logger.info(f"[LLM] Creating chat with prompt size: {len(prompt)} chars")
        chat = await self.coze.chat.create(
            bot_id=self.bot_id,
            user_id=self.user_id,
            additional_messages=[Message.build_user_question_text(prompt)],
        )
        logger.info(f"[LLM] Chat created: {chat.id}, status: {chat.status}")
        
        # Combine assistant messages
        # Poll until complete with async sleep
        wait_count = 0
        while chat.status == ChatStatus.IN_PROGRESS:
            wait_count += 1
            if wait_count % 10 == 0:  # Log every 10 seconds
                logger.info(f"[LLM] Still waiting... ({wait_count}s)")
            await asyncio.sleep(1)  # Use asyncio.sleep, not time.sleep  
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
            print(f"Error retrieving messages: {e}")
            # Try to get the chat result directly if message retrieval fails
            if hasattr(chat, 'last_error') and chat.last_error:
                return f"Error: {chat.last_error.msg}"
            return "Error retrieving response from AI. Please try a simpler query."
      
        # Filter for answer messages, handling potential validation errors
        answer_messages = []
        for msg in messages:
            try:
                if hasattr(msg, 'type') and msg.type == MessageType.ANSWER:
                    answer_messages.append(msg)
            except Exception as e:
                # Skip messages that don't validate properly (e.g., code execution messages)
                print(f"Skipping message due to validation error: {e}")
                continue 
  
        if answer_messages:  
            # Get the first answer (usually there's only one)
            content = answer_messages[0].content
            try:
                # Try to parse as JSON first
                data = json.loads(content)
                answer = data.get("response", content)
            except (json.JSONDecodeError, TypeError):
                # If not JSON, use content directly
                answer = content
        else:
            answer = ""
        # Normalize: if answer is a JSON string with {"response": ...}, extract value
        if isinstance(answer, str):
            try:
                obj = json.loads(answer)
                if isinstance(obj, dict) and "response" in obj:
                    return str(obj["response"])
            except Exception:
                pass
        return answer

    
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
    GET /v1/async_result?task_id=... -> polls until {"status_code": 200, "data": {...}, "err": ""}
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
            # Allow up to 100 concurrent connections for high student load (30-60 users)
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def ainvoke(self, prompt: str) -> str:
        """Submit async chat request and poll for result."""
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
            await asyncio.sleep(1.0 if poll_count < 10 else 2.0)  # Fast first, then slower
            
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
                
                # Log progress every 10s
                current_time = asyncio.get_event_loop().time()
                elapsed = current_time - start_time
                if current_time - last_log_time >= 10.0:
                    logger.info(f"[302] Polling... {elapsed:.1f}s elapsed (poll #{poll_count}, status: {status_code}, err: {err})")
                    last_log_time = current_time
                
                # Check if done: status_code 200, has data, err is empty or not "pending"
                if status_code == 200 and data and err not in ("result pending", "pending"):
                    logger.info(f"[302] Completed in {elapsed:.1f}s")
                    # Extract text from data
                    return self._extract_text(data)
                
                # Timeout after 5 minutes
                if elapsed > 300:
                    logger.warning(f"[302] Timeout after {elapsed:.0f}s")
                    raise TimeoutError(f"Async result timeout. Last: {result}")
    
    def _extract_response_from_content_str(self, content: str) -> str:
        """Try to parse content as JSON with 'response' field, or extract it from
        partial JSON-like strings; otherwise return content as-is.
        """
        # 1) Full JSON like {"response": "..."}
        try:
            obj = json.loads(content)
            if isinstance(obj, dict) and "response" in obj:
                return str(obj["response"])
        except (json.JSONDecodeError, TypeError):
            pass

        # 2) Fallback: handle fragments such as '"response": "..."}'
        # or similar JSON-like snippets that are not valid standalone JSON.
        try:
            import re
            m = re.search(r'"response"\s*:\s*"([\s\S]*?)"\s*}?$', content)
            if not m:
                m = re.search(r'"response"\s*:\s*"([\s\S]*?)"', content)
            if m:
                inner = m.group(1)
                # Decode common JSON escape sequences (\n, \", \\) if present.
                try:
                    return json.loads(f'"{inner}"')
                except Exception:
                    return inner
        except Exception:
            pass

        return content
    
    def _extract_text(self, data) -> str:
        """Extract final text from the data field per docs."""
        # Case 1: data is a dict (similar to OpenAI response)
        if isinstance(data, dict):
            # Try OpenAI-like choices
            if "choices" in data:
                try:
                    msg = data["choices"][0].get("message") or {}
                    content = msg.get("content")
                    if isinstance(content, str):
                        logger.info(f"[302] Found content in choices: {content[:100]}...")
                        result = self._extract_response_from_content_str(content)
                        logger.info(f"[302] Extracted: {result}")
                        return result
                except Exception as e:
                    logger.info(f"[302] Error extracting from choices: {e}")
                    pass
            # Direct response field
            if "response" in data:
                logger.info(f"[302] Found direct 'response' field")
                return str(data["response"])
            # Fallback to JSON dump
            logger.info(f"[302] Fallback: returning JSON dump")
            return json.dumps(data, ensure_ascii=False)

        # Case 2: data is a string (raw JSON or plain text)
        if isinstance(data, str):
            # Try to parse as full OpenAI-like response
            try:
                parsed = json.loads(data)
                if isinstance(parsed, dict):
                    logger.info(f"[302] Parsed JSON string into dict")
                    if "choices" in parsed:
                        try:
                            msg = parsed["choices"][0].get("message") or {}
                            content = msg.get("content")
                            if isinstance(content, str):
                                logger.info(f"[302] Found content in parsed choices: {content[:100]}...")
                                result = self._extract_response_from_content_str(content)
                                logger.info(f"[302] Extracted: {result}")
                                return result
                        except Exception as e:
                            logger.info(f"[302] Error extracting from parsed choices: {e}")
                            pass
                    if "response" in parsed:
                        logger.info(f"[302] Found 'response' in parsed dict")
                        return str(parsed["response"])
            except (json.JSONDecodeError, TypeError) as e:
                logger.info(f"[302] Not valid JSON: {e}")
                pass
            # Or parse as {"response": ...}
            logger.info(f"[302] Trying to extract response from content string")
            return self._extract_response_from_content_str(data)

        # Unknown type -> stringify
        logger.info(f"[302] Unknown data type: {type(data)}")
        return "" if data is None else str(data)

    async def aclose(self):
        if self._session and not self._session.closed:
            await self._session.close()


def load_302_llm() -> Optional[A302LLM]:
    """Factory: returns a configured 302.ai LLM or None if env is missing."""
    if not A302_API_KEY:
        return None
    return A302LLM(api_key=A302_API_KEY, base_url=A302_API_BASE, model_name=A302_MODEL_NAME)