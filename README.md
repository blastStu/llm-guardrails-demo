# LLM Guardrail Demonstrations

This project contains demonstration code for applying input guardrails to LLM inference calls. It is **not production code** — the examples are intentionally simple to illustrate the concepts and flow clearly.

The core idea: before sending a user prompt to an LLM for inference, run a guardrail check to detect prompt injection attempts. Only if the check passes does the actual inference call proceed. The same check is applied to tool results returned to the model — a common indirect prompt injection vector, where an attacker embeds instructions in external data (an email, a document, a web page) that the model will read.

**Scope:** these guardrails check specifically for prompt injection — attempts to hijack or override the model's behaviour through crafted input. They do not check for off-topic requests, harmful content, policy violations, PII, or other categories. NeMo Guardrails supports those rail types, but they are not configured here.

---

## Architecture

The guardrail check is an LLM-as-judge pattern:

1. Take the user prompt
2. Insert it into a classification prompt template that asks the judge LLM: *"Does this contain a prompt injection? Yes or No?"*
3. Parse the first two words of the judge's response (`yes`/`no`/`safe`/`unsafe`)
4. If blocked → return early, no inference call made
5. If passed → send the prompt to the LLM for actual inference

The judge LLM and the inference LLM are the same endpoint in these demos, but they could be different models.

---

## Environment

All implementations read from a `.env` file. Copy `.env.example` to `.env` and fill in your values:

```
OPENAICOMPATIBLE_API_KEY=sk-...          # API key for the upstream model endpoint
UPSTREAM_BASE_URL=https://...            # base URL of the upstream OpenAI-compatible endpoint
HTTPS_PROXY=http://proxy-host:port       # corporate HTTP proxy if required
HTTP_PROXY=http://proxy-host:port
```

The upstream endpoint must be an OpenAI-compatible API, configured via `UPSTREAM_BASE_URL` in `.env`. The examples use the model `rits/zai-org/glm-5-2-fp8` but any model served by the upstream will work.

---

## Files

### `inference.py` — Python, NeMo Guardrails

The primary Python demonstration. Uses the `nemoguardrails` library to run the guardrail check, then calls the LLM directly using the `openai` Python client.

**Flow:**
1. Load NeMo config from `config/`
2. Call `rails.check()` with `RailType.INPUT` — NeMo sends the classification prompt to the judge LLM and parses the result
3. If `RailStatus.BLOCKED` → stop
4. If passed → call the LLM via `client.chat.completions.create()`

**Key point:** `rails.check()` with `RailType.INPUT` only runs the guardrail classification — it does **not** call the LLM for inference. The inference call is made separately and explicitly in step 4. This gives full control over the inference call (parameters, model, streaming, etc.).

**To run:**
```bash
uv sync
python inference.py
```

---

### `config/config.yaml` — NeMo Guardrails configuration

Configures the NeMo rail that `inference.py` uses. Contains:
- The LLM endpoint and model for the judge
- The rail type (`self check input`)
- The classification prompt template (the `prompts` section)

The classification prompt is a prompt injection detector with a delimiter trick (`===!@#$%^===`) to prevent the user input from breaking out of the template. The output parser is `is_content_safe`, which checks the first two words of the judge's response for `yes`/`no`/`safe`/`unsafe` and defaults to blocking if the response is ambiguous.

---

### `inference_tools.py` — Python, NeMo Guardrails, with tool calls

Extends `inference.py` to demonstrate guardrail checking in a tool-calling flow. The scenario: a user asks an AI assistant to summarise an email. The assistant calls a `get_email` tool. The email content may contain a prompt injection planted by an attacker.

**Flow:**
1. Check the user message with NeMo (same as `inference.py`)
2. Send to LLM with tool definitions — LLM responds with a tool call
3. Execute the tool (mocked email inbox — `email-001` is clean, `email-002` contains an injection)
4. **Check the tool result with NeMo before adding it to the conversation** — this is the key step, catching indirect prompt injection from external data sources
5. If the tool result passes → add it to conversation history and call LLM for final response

**Implementation note:** NeMo's `check()` API only inspects the last `user` message in a conversation. Tool results have `role: "tool"` and are not checked automatically. The workaround used here is to wrap the tool result in a fake `user` message for the NeMo check, then add the real `role: "tool"` message to the history once it has passed. This is a pragmatic approach given the current NeMo API.

**To run:**
```bash
python inference_tools.py
# Change "email-002" to "email-001" in the user message to see the non-blocked path
```

---


### `nemo_guardrail.py` + `litellm_config.yaml` — LiteLLM proxy with NeMo Guardrails

The most complete demonstration: a LiteLLM proxy that intercepts every incoming request and runs NeMo Guardrail checks before forwarding to the upstream model. Clients talk to the proxy over a standard OpenAI-compatible API — they need no knowledge of the guardrail layer.

**Overview:**
- `litellm_config.yaml` configures the proxy: wildcard model routing to the upstream endpoint, and registration of the NeMo guardrail plugin
- `nemo_guardrail.py` is a LiteLLM `CustomGuardrail` that implements `async_pre_call_hook()` — it runs before every request reaches the upstream
- The hook checks the last user message and every `role: "tool"` message (indirect prompt injection vector) against NeMo's input rail
- Blocked requests receive a synthetic `"I'm sorry, I can't help with that."` chat completion response (HTTP 200) — the upstream is never called
- Guardrail errors (judge LLM unreachable, exception during check) return HTTP 503, so automated tools can distinguish a deliberate block from a failed safety check

**Why this approach over a hand-rolled gateway:**  
LiteLLM handles the OpenAI-compatible API surface, model routing, streaming, retries, and observability. The custom code is only the guardrail logic itself — roughly 50 lines in `nemo_guardrail.py`. Everything else is configuration.

**This is a demo implementation.** It has not been load-tested or hardened for production. See [QUICKSTART.md](QUICKSTART.md) for setup, curl examples, and a full list of limitations to be aware of before using this in any serious context.

**To run:**
```bash
uv sync
litellm --config litellm_config.yaml
```

Or with Docker:
```bash
docker build -t nemo-litellm-proxy .
docker run --env-file .env -p 4000:4000 nemo-litellm-proxy
```

See [QUICKSTART.md](QUICKSTART.md) for the full guide.

---

## What is NeMo Guardrails?

NeMo Guardrails is an open-source NVIDIA library (`nemoguardrails` on PyPI) for adding configurable guardrail checks to LLM applications. It supports multiple rail types beyond LLM-as-judge, including regex, Llama Guard, PII detection, and others. Rails are configured via YAML and Colang (a domain-specific language for dialogue flows).

The `self check input` rail used here is the simplest built-in rail: it sends a classification prompt to the LLM and parses the response. The prompt template is defined in `config/config.yaml` and can be customised to enforce any policy.

Current release: `0.23.0`. Source: https://github.com/NVIDIA/NeMo-Guardrails

---

## What these demos do NOT cover

- Output rails (checking the LLM's response before returning it to the user)
- Multi-turn conversation guardrailing beyond the last user message
- Non-LLM rail types (regex, Llama Guard, PII detection)
- Streaming responses
- Error handling beyond the happy path
- Production deployment, scaling, or latency considerations
