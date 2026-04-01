"""
bot.py — Reflection-bot Telegram interface.

State machine for the three-phase reflection flow:

  INIT
    ↓ /start or any message
  P1_GOALS  → P1_ACTIONS  → P1_RESULTS  → P1_AWAIT_CONFIRM
    ↓ any message
  P2_EPISODE → P2_ACTIONS → P2_INTERACTION → P2_THINKING → P2_AWAIT_CONFIRM
    ↓ any message
  P3_UNDERSTANDING → P3_INTERACTION → P3_ACTIONS → DONE

State transitions are driven by the `layer_complete` flag returned
by each skill function in skills.py.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field

import db
from enum import Enum
from typing import List, Optional, Tuple

from telebot.async_telebot import AsyncTeleBot

from model_loading import load_coze_llm, load_302_llm
from settings import PROJECT_ROOT, TELEGRAM_BOT_TOKEN
from skills import (
    p1_goals, p1_actions, p1_results, p1_summary,
    p2_episode_select, p2_actions, p2_interaction, p2_thinking, p2_summary,
    p3_understanding, p3_interaction, p3_actions, p3_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN is required.")


# ─────────────────────────────────────────────────────────────────────────────
# State definitions
# ─────────────────────────────────────────────────────────────────────────────

class BotState(str, Enum):
    """All possible states in the reflection flow."""
    INIT               = "init"
    # Phase 1 – Situation Analysis
    P1_GOALS           = "p1_goals"
    P1_ACTIONS         = "p1_actions"
    P1_RESULTS         = "p1_results"
    P1_AWAIT_CONFIRM   = "p1_await_confirm"   # summary shown, waiting for "continue"
    # Phase 2 – Reflective Action
    P2_EPISODE         = "p2_episode"
    P2_ACTIONS         = "p2_actions"
    P2_INTERACTION     = "p2_interaction"
    P2_THINKING        = "p2_thinking"
    P2_AWAIT_CONFIRM   = "p2_await_confirm"   # summary shown, waiting for "continue"
    # Phase 3 – Designing New Action
    P3_UNDERSTANDING   = "p3_understanding"
    P3_INTERACTION     = "p3_interaction"
    P3_ACTIONS         = "p3_actions"
    DONE               = "done"


@dataclass
class UserSession:
    """Per-user persistent state across the full reflection flow."""
    state: BotState = BotState.INIT

    # ── Phase 1 accumulated data ──────────────────────────────────────────
    p1_goals_data: str = ""
    p1_actions_data: str = ""
    p1_results_data: str = ""
    p1_summary_text: str = ""

    # ── Phase 2 accumulated data ──────────────────────────────────────────
    episode: str = ""
    p2_action_gap: str = ""
    p2_interaction_gap: str = ""
    p2_thinking_gap: str = ""
    p2_summary_text: str = ""

    # ── Phase 3 accumulated data ──────────────────────────────────────────
    p3_new_understanding: str = ""
    p3_new_participation: str = ""
    p3_new_actions: str = ""

    # ── DB session tracking ────────────────────────────────────────────────
    session_db_id: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Chat history (per chat_id)
# ─────────────────────────────────────────────────────────────────────────────

class ChatHistory:
    """Maintains a rolling window of (role, text) pairs per chat."""

    def __init__(self, max_turns: int = 20):
        self._max = max_turns
        self._store: dict[int, List[Tuple[str, str]]] = {}

    def add(self, chat_id: int, role: str, text: str) -> None:
        turns = self._store.setdefault(chat_id, [])
        turns.append((role, text))
        if len(turns) > self._max:
            self._store[chat_id] = turns[-self._max:]

    def format(self, chat_id: int) -> str:
        turns = self._store.get(chat_id, [])
        if not turns:
            return ""
        lines = []
        for role, text in turns:
            prefix = "Студент:" if role == "user" else "Бот:"
            lines.append(f"{prefix} {text}")
        return "\n".join(lines)

    def clear(self, chat_id: int) -> None:
        self._store.pop(chat_id, None)


# ─────────────────────────────────────────────────────────────────────────────
# Message sender with basic retry
# ─────────────────────────────────────────────────────────────────────────────

class MessageSender:
    """Sends messages with exponential-backoff retry."""

    MAX_LEN = 4000
    MAX_ATTEMPTS = 3

    def __init__(self, bot: AsyncTeleBot):
        self._bot = bot

    async def send(self, chat_id: int, text: str) -> None:
        """Send text, splitting into chunks if necessary."""
        chunks = [text[i:i + self.MAX_LEN] for i in range(0, len(text), self.MAX_LEN)]
        for chunk in chunks:
            await self._send_with_retry(chat_id, chunk)

    async def _send_with_retry(self, chat_id: int, text: str) -> None:
        last_err: Exception | None = None
        for attempt in range(self.MAX_ATTEMPTS):
            try:
                await self._bot.send_message(chat_id, text, timeout=60)
                return
            except Exception as exc:
                last_err = exc
                logger.warning("send_message attempt %d/%d failed: %s",
                               attempt + 1, self.MAX_ATTEMPTS, exc)
                await asyncio.sleep(2 ** attempt)
        logger.error("Failed to send message after %d attempts: %s",
                     self.MAX_ATTEMPTS, last_err)


# ─────────────────────────────────────────────────────────────────────────────
# Main bot
# ─────────────────────────────────────────────────────────────────────────────

WELCOME_TEXT = (
    "Привет! Я помогу тебе пройти через рефлексию учебной недели.\n\n"
    "Мы работаем в три этапа:\n"
    "1️⃣ Восстановим неделю — задачи, действия, результаты.\n"
    "2️⃣ Разберём один эпизод в трёх слоях: деятельность, взаимодействие, мышление.\n"
    "3️⃣ Сконструируем новый способ действия.\n\n"
    "Всё, что ты напишешь, остаётся между нами и служит только для рефлексии. "
    "Пиши свободно.\n\n"
    "Давай начнём. С какими задачами и целями ты входил в эту неделю?"
)

RESTART_TEXT = (
    "Сессия рефлексии сброшена. Начинаем сначала.\n\n"
    "С какими задачами и целями ты входил в эту неделю?"
)

AWAIT_CONFIRM_SUFFIX = "\n\nНапиши что-нибудь, когда будешь готов продолжить."


class ReflectionBot:
    """Core bot: routes each incoming message to the appropriate skill."""

    def __init__(self, token: str, llm):
        self._bot = AsyncTeleBot(token)
        self._llm = llm
        self._sender = MessageSender(self._bot)
        self._history = ChatHistory(max_turns=30)
        self._sessions: dict[int, UserSession] = {}   # user_id → session
        self._active: dict[int, int] = {}             # user_id → active request count

    # ── Session helpers ───────────────────────────────────────────────────

    def _session(self, user_id: int) -> UserSession:
        if user_id not in self._sessions:
            self._sessions[user_id] = UserSession()
        return self._sessions[user_id]

    def _reset(self, user_id: int, chat_id: int) -> None:
        self._sessions[user_id] = UserSession()
        self._history.clear(chat_id)

    # ── Concurrency guard ─────────────────────────────────────────────────

    def _busy(self, user_id: int) -> bool:
        return self._active.get(user_id, 0) > 0

    def _enter(self, user_id: int) -> None:
        self._active[user_id] = self._active.get(user_id, 0) + 1

    def _leave(self, user_id: int) -> None:
        self._active[user_id] = max(0, self._active.get(user_id, 0) - 1)
        if self._active[user_id] == 0:
            del self._active[user_id]

    # ── DB logging helpers ────────────────────────────────────────────────

    def _make_session_id(self, user_id: int) -> str:
        return f"{user_id}_{int(time.time())}"

    async def _log_msg(self, user_id: int, role: str, text: str, state: str = "") -> None:
        session = self._session(user_id)
        if session.session_db_id:
            await db.log_turn(session.session_db_id, user_id, role, text, state)

    # ── Message routing ───────────────────────────────────────────────────

    async def _route(self, user_id: int, chat_id: int, text: str) -> None:
        """Dispatch a message to the correct skill based on current state."""
        session = self._session(user_id)
        history = self._history.format(chat_id)

        result: dict | None = None

        # ── Phase 1 ──────────────────────────────────────────────────────
        if session.state == BotState.P1_GOALS:
            result = await p1_goals(self._llm, text, history)
            if result.get("layer_complete"):
                session.p1_goals_data = result.get("layer_data") or ""
                session.state = BotState.P1_ACTIONS

        elif session.state == BotState.P1_ACTIONS:
            result = await p1_actions(self._llm, text, session.p1_goals_data, history)
            if result.get("layer_complete"):
                session.p1_actions_data = result.get("layer_data") or ""
                session.state = BotState.P1_RESULTS

        elif session.state == BotState.P1_RESULTS:
            result = await p1_results(
                self._llm, text,
                session.p1_goals_data, session.p1_actions_data,
                history,
            )
            if result.get("layer_complete"):
                session.p1_results_data = result.get("layer_data") or ""
                # Auto-generate phase 1 summary immediately
                await self._sender.send(chat_id, result["response"])
                self._history.add(chat_id, "user", text)
                self._history.add(chat_id, "assistant", result["response"])
                await self._log_msg(user_id, "bot", result["response"], session.state.value)
                summary_result = await p1_summary(
                    self._llm,
                    session.p1_goals_data,
                    session.p1_actions_data,
                    session.p1_results_data,
                    self._history.format(chat_id),
                )
                if summary_result.get("layer_complete"):
                    session.p1_summary_text = summary_result.get("layer_data") or ""
                    await db.update_session(session.session_db_id, p1_summary=session.p1_summary_text)
                    session.state = BotState.P1_AWAIT_CONFIRM
                    final_text = summary_result["response"] + AWAIT_CONFIRM_SUFFIX
                    await self._sender.send(chat_id, final_text)
                    self._history.add(chat_id, "assistant", summary_result["response"])
                    await self._log_msg(user_id, "bot", final_text, session.state.value)
                else:
                    # Summary formation failed — ask for more info
                    await self._sender.send(chat_id, summary_result["response"])
                    self._history.add(chat_id, "assistant", summary_result["response"])
                    await self._log_msg(user_id, "bot", summary_result["response"], session.state.value)
                    # Stay in P1_RESULTS
                return
            # layer not yet complete — fall through to normal send below

        # ── Await confirm → Phase 2 ───────────────────────────────────────
        elif session.state == BotState.P1_AWAIT_CONFIRM:
            session.state = BotState.P2_EPISODE
            result = await p2_episode_select(
                self._llm, text, session.p1_summary_text, history
            )
            if result.get("layer_complete"):
                session.episode = result.get("layer_data") or ""
                session.state = BotState.P2_ACTIONS

        # ── Phase 2 ──────────────────────────────────────────────────────
        elif session.state == BotState.P2_EPISODE:
            result = await p2_episode_select(
                self._llm, text, session.p1_summary_text, history
            )
            if result.get("layer_complete"):
                session.episode = result.get("layer_data") or ""
                session.state = BotState.P2_ACTIONS

        elif session.state == BotState.P2_ACTIONS:
            result = await p2_actions(self._llm, text, session.episode, history)
            if result.get("layer_complete"):
                session.p2_action_gap = result.get("layer_data") or ""
                session.state = BotState.P2_INTERACTION

        elif session.state == BotState.P2_INTERACTION:
            result = await p2_interaction(
                self._llm, text, session.episode, session.p2_action_gap, history
            )
            if result.get("layer_complete"):
                session.p2_interaction_gap = result.get("layer_data") or ""
                session.state = BotState.P2_THINKING

        elif session.state == BotState.P2_THINKING:
            result = await p2_thinking(
                self._llm, text, session.episode,
                session.p2_action_gap, session.p2_interaction_gap,
                history,
            )
            if result.get("layer_complete"):
                session.p2_thinking_gap = result.get("layer_data") or ""
                # Auto-generate phase 2 summary
                await self._sender.send(chat_id, result["response"])
                self._history.add(chat_id, "user", text)
                self._history.add(chat_id, "assistant", result["response"])
                await self._log_msg(user_id, "bot", result["response"], session.state.value)
                summary_result = await p2_summary(
                    self._llm,
                    session.episode,
                    session.p2_action_gap,
                    session.p2_interaction_gap,
                    session.p2_thinking_gap,
                    self._history.format(chat_id),
                )
                if summary_result.get("layer_complete"):
                    session.p2_summary_text = summary_result.get("layer_data") or ""
                    await db.update_session(session.session_db_id, p2_summary=session.p2_summary_text)
                    session.state = BotState.P2_AWAIT_CONFIRM
                    final_text = summary_result["response"] + AWAIT_CONFIRM_SUFFIX
                    await self._sender.send(chat_id, final_text)
                    self._history.add(chat_id, "assistant", summary_result["response"])
                    await self._log_msg(user_id, "bot", final_text, session.state.value)
                else:
                    await self._sender.send(chat_id, summary_result["response"])
                    self._history.add(chat_id, "assistant", summary_result["response"])
                    await self._log_msg(user_id, "bot", summary_result["response"], session.state.value)
                    # Stay in P2_THINKING for another pass
                return

        # ── Await confirm → Phase 3 ───────────────────────────────────────
        elif session.state == BotState.P2_AWAIT_CONFIRM:
            session.state = BotState.P3_UNDERSTANDING
            result = await p3_understanding(
                self._llm, text,
                session.episode, session.p2_summary_text,
                history,
            )
            if result.get("layer_complete"):
                session.p3_new_understanding = result.get("layer_data") or ""
                session.state = BotState.P3_INTERACTION

        # ── Phase 3 ──────────────────────────────────────────────────────
        elif session.state == BotState.P3_UNDERSTANDING:
            result = await p3_understanding(
                self._llm, text,
                session.episode, session.p2_summary_text,
                history,
            )
            if result.get("layer_complete"):
                session.p3_new_understanding = result.get("layer_data") or ""
                session.state = BotState.P3_INTERACTION

        elif session.state == BotState.P3_INTERACTION:
            result = await p3_interaction(
                self._llm, text,
                session.episode, session.p3_new_understanding,
                history,
            )
            if result.get("layer_complete"):
                session.p3_new_participation = result.get("layer_data") or ""
                session.state = BotState.P3_ACTIONS

        elif session.state == BotState.P3_ACTIONS:
            result = await p3_actions(
                self._llm, text,
                session.episode,
                session.p3_new_understanding,
                session.p3_new_participation,
                history,
            )
            if result.get("layer_complete"):
                session.p3_new_actions = result.get("layer_data") or ""
                # Auto-generate final summary
                await self._sender.send(chat_id, result["response"])
                self._history.add(chat_id, "user", text)
                self._history.add(chat_id, "assistant", result["response"])
                await self._log_msg(user_id, "bot", result["response"], session.state.value)
                final_result = await p3_summary(
                    self._llm,
                    session.episode,
                    session.p3_new_understanding,
                    session.p3_new_participation,
                    session.p3_new_actions,
                    self._history.format(chat_id),
                )
                final_text = final_result["response"]
                if final_result.get("layer_complete"):
                    session.state = BotState.DONE
                    final_text += (
                        "\n\n✅ Сессия рефлексии завершена. "
                        "Если хочешь начать новую сессию, введи /start."
                    )
                    await db.update_session(
                        session.session_db_id,
                        p3_summary=final_result.get("layer_data") or "",
                    )
                    await db.close_session(session.session_db_id, BotState.DONE.value)
                else:
                    # Coherence check failed — need to revisit a layer
                    # state stays at P3_ACTIONS; bot will prompt what to fix
                    pass
                await self._sender.send(chat_id, final_text)
                self._history.add(chat_id, "assistant", final_text)
                await self._log_msg(user_id, "bot", final_text, session.state.value)
                return

        # ── DONE state ────────────────────────────────────────────────────
        elif session.state == BotState.DONE:
            await self._sender.send(
                chat_id,
                "Рефлексия уже завершена. Чтобы начать новую сессию, введи /start.",
            )
            return

        # ── INIT fallback (should not happen after /start, but just in case)
        else:
            session.state = BotState.P1_GOALS
            result = await p1_goals(self._llm, text, history)
            if result.get("layer_complete"):
                session.p1_goals_data = result.get("layer_data") or ""
                session.state = BotState.P1_ACTIONS

        # ── Send result ───────────────────────────────────────────────────
        if result:
            response_text = result.get("response", "")
            if response_text:
                await self._sender.send(chat_id, response_text)
                self._history.add(chat_id, "user", text)
                self._history.add(chat_id, "assistant", response_text)
                await self._log_msg(user_id, "bot", response_text, session.state.value)

    # ── Telegram handlers ─────────────────────────────────────────────────

    async def handle_start(self, message) -> None:
        user_id = message.from_user.id
        chat_id = message.chat.id
        self._reset(user_id, chat_id)
        session = self._session(user_id)
        session.state = BotState.P1_GOALS
        session.session_db_id = self._make_session_id(user_id)
        await db.create_session(session.session_db_id, user_id)
        await self._sender.send(chat_id, WELCOME_TEXT)
        self._history.add(chat_id, "assistant", WELCOME_TEXT)
        await self._log_msg(user_id, "bot", WELCOME_TEXT, BotState.P1_GOALS.value)

    async def handle_restart(self, message) -> None:
        user_id = message.from_user.id
        chat_id = message.chat.id
        self._reset(user_id, chat_id)
        session = self._session(user_id)
        session.state = BotState.P1_GOALS
        session.session_db_id = self._make_session_id(user_id)
        await db.create_session(session.session_db_id, user_id)
        await self._sender.send(chat_id, RESTART_TEXT)
        self._history.add(chat_id, "assistant", RESTART_TEXT)
        await self._log_msg(user_id, "bot", RESTART_TEXT, BotState.P1_GOALS.value)

    async def handle_message(self, message) -> None:
        user_id = message.from_user.id
        chat_id = message.chat.id
        text = (message.text or "").strip()

        if not text:
            return

        if self._busy(user_id):
            await self._sender.send(
                chat_id, "⏳ Обрабатываю предыдущее сообщение, подожди немного."
            )
            return

        session = self._session(user_id)

        # Auto-start if the user writes without /start
        if session.state == BotState.INIT:
            session.state = BotState.P1_GOALS
            if not session.session_db_id:
                session.session_db_id = self._make_session_id(user_id)
                await db.create_session(session.session_db_id, user_id)

        # Log incoming user message
        await self._log_msg(user_id, "user", text, session.state.value)

        async def _keep_typing():
            while True:
                try:
                    await self._bot.send_chat_action(chat_id, "typing")
                    await asyncio.sleep(4)
                except asyncio.CancelledError:
                    break

        self._enter(user_id)
        typing_task = asyncio.create_task(_keep_typing())
        try:
            await self._route(user_id, chat_id, text)
        except Exception as exc:
            logger.error("Error processing message for user %s: %s", user_id, exc, exc_info=True)
            await self._sender.send(
                chat_id,
                "❗ Произошла ошибка. Попробуй ещё раз или перезапусти сессию командой /restart.",
            )
        finally:
            typing_task.cancel()
            self._leave(user_id)

    # ── Polling loop ──────────────────────────────────────────────────────

    async def _polling_loop(self) -> None:
        logger.info("Starting Telegram long-polling loop.")
        offset: int | None = None

        while True:
            try:
                updates = await self._bot.get_updates(
                    offset=offset, limit=100, timeout=20
                )
            except Exception as exc:
                logger.error("get_updates error: %s", exc)
                await asyncio.sleep(5)
                continue

            for upd in updates:
                offset = upd.update_id + 1
                msg = upd.message
                if not msg or msg.content_type != "text":
                    continue
                text = (msg.text or "").strip()
                if text.startswith("/start"):
                    asyncio.create_task(self.handle_start(msg))
                elif text.startswith("/restart"):
                    asyncio.create_task(self.handle_restart(msg))
                else:
                    asyncio.create_task(self.handle_message(msg))

    async def run(self) -> None:
        try:
            await self._polling_loop()
        except KeyboardInterrupt:
            logger.info("Bot shutdown requested.")
        except Exception:
            logger.exception("Critical bot error.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    await db.init_db()
    provider = os.getenv("LLM_PROVIDER", "coze").lower()
    if provider == "302":
        llm = load_302_llm()
    else:
        llm = load_coze_llm()

    if llm is None:
        raise ValueError(
            "LLM failed to load. Check LLM_PROVIDER and related env variables."
        )
    logger.info("LLM loaded (%s).", provider)

    bot = ReflectionBot(TELEGRAM_BOT_TOKEN, llm)
    try:
        await bot.run()
    finally:
        await db.close_db()


if __name__ == "__main__":
    asyncio.run(main())