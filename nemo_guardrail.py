import asyncio
import logging
import os

from fastapi import HTTPException
from litellm.integrations.custom_guardrail import CustomGuardrail, ModifyResponseException
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.options import RailStatus, RailType

logging.getLogger("nemoguardrails").setLevel(logging.WARNING)
logging.getLogger("nemoguardrails.actions.llm.utils").setLevel(logging.INFO)
logging.getLogger("nemoguardrails.library.self_check").setLevel(logging.INFO)

log = logging.getLogger(__name__)

_GUARDRAIL_CONCURRENCY = int(os.environ.get("GUARDRAIL_CONCURRENCY", "4"))


class NemoGuardrail(CustomGuardrail):
    def __init__(self, rails_config_path: str = "./config", **kwargs):
        super().__init__(**kwargs)
        config = RailsConfig.from_path(rails_config_path)
        config.models[0].api_key_env_var = "OPENAICOMPATIBLE_API_KEY"
        config.models[0].parameters["base_url"] = os.environ["UPSTREAM_BASE_URL"]
        self._rails = LLMRails(config)
        self._sem = asyncio.Semaphore(_GUARDRAIL_CONCURRENCY)
        log.info(
            "NemoGuardrail ready (config=%s, max_concurrent=%d)",
            rails_config_path,
            _GUARDRAIL_CONCURRENCY,
        )

    async def _check_content(self, content: str, model: str, data: dict, label: str) -> None:
        """Run a NeMo input check on a single piece of content, blocking if flagged."""
        fake_messages = [{"role": "user", "content": content}]
        try:
            async with self._sem:
                result = await self._rails.check_async(
                    messages=fake_messages, rail_types=[RailType.INPUT]
                )
        except Exception:
            log.exception("NeMo guardrail check failed on %s.", label)
            raise HTTPException(
                status_code=503,
                detail="Guardrail check unavailable — request blocked.",
            )

        if result.status == RailStatus.BLOCKED:
            log.info("Content blocked by rail on %s: %s", label, result.rail)
            raise ModifyResponseException(
                message="I'm sorry, I can't help with that.",
                model=model,
                request_data=data,
                guardrail_name=self.guardrail_name,
                detection_info={"rail": result.rail or "self check input", "source": label},
            )

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        messages = data.get("messages", [])
        model = data.get("model", "unknown")

        # Step 1: check the last user message (NeMo's built-in behaviour).
        await self._check_content(
            content=next(
                (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
                "",
            ),
            model=model,
            data=data,
            label="user message",
        )

        # Step 2: check each tool result — NeMo skips role:tool messages by default,
        # but tool results are a primary indirect prompt injection vector.
        for i, msg in enumerate(messages):
            if msg.get("role") != "tool":
                continue
            content = msg.get("content") or ""
            # Tool content may be a list of content blocks; flatten to plain text.
            if isinstance(content, list):
                content = " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
            if not content.strip():
                continue
            await self._check_content(
                content=f"Tool returned: {content}",
                model=model,
                data=data,
                label=f"tool message [{i}]",
            )

        return data
