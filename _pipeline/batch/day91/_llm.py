"""Provider shim so the short-video pipeline can run without Anthropic credits.

Anthropic ran out of credits, which stops this pipeline dead: every script, title and case
pick goes through claude-sonnet-5. Fireworks is the fallback, and the comic pipeline has
already proven glm-5p2 on the same kind of work.

WHY A SHIM AND NOT A REWRITE. All three callers consume the response the same way:

    raw = next((b.text for b in resp.content if b.type == "text"), None)
    if resp.stop_reason == "max_tokens": ...

That retry-on-truncation logic is correct and hard-won -- _gen_video_content hit max_tokens
on 3/3 attempts for one case before its budget was raised. Rewriting each call site to a
different response shape would mean re-deriving that logic three times. So this returns an
object with the same surface instead, and each call site changes by exactly one line:

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])   ->   client = _llm.client()

Set SHORT_LLM_PROVIDER=anthropic to switch back when credits are topped up. The default is
"auto": use Anthropic when its key is present, otherwise Fireworks. Auto is the default
rather than a hardcoded provider because the failure this exists to fix was a MISSING key
taking down a scheduled run -- a pipeline that cannot reach one provider should move to the
other, not stop.

NOT FOR VISION. The image QA inside the Kaggle kernels sends base64 images, and glm-5p2 is
text-only. Passing image blocks here raises rather than silently dropping them, because the
kernel's QA already treats a failed vision call as PASS -- a dropped image would turn that
into "every image passes", which looks exactly like a working QA gate that has stopped
checking anything.
"""
import json
import os
import time
import urllib.error
import urllib.request

FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
FIREWORKS_MODEL = os.environ.get("FIREWORKS_MODEL",
                                 "accounts/fireworks/models/glm-5p2")

# Reasoning models will think past any budget they are given: on the comic script prompt
# glm-5p2 produced 102,699 characters of reasoning and still returned no content at a
# 32,000-token cap. Capping the effort is what makes the call terminate at all.
REASONING_EFFORT = os.environ.get("FIREWORKS_REASONING_EFFORT", "low")


class TextBlock:
    """Mimics an Anthropic content block: `.type` and `.text`."""

    type = "text"

    def __init__(self, text):
        self.text = text


class Response:
    """Mimics an Anthropic message: `.content` (blocks) and `.stop_reason`."""

    def __init__(self, text, stop_reason):
        self.content = [TextBlock(text)]
        self.stop_reason = stop_reason


def _flatten(messages):
    """Anthropic messages -> OpenAI messages.

    Anthropic allows content to be a list of typed blocks; the chat-completions dialect
    wants a plain string. Text blocks are joined; anything else is refused loudly (see the
    module docstring on why silence would be worse).
    """
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            kinds = {b.get("type") for b in content if isinstance(b, dict)}
            if kinds - {"text"}:
                raise ValueError(
                    f"_llm shim received non-text content blocks ({sorted(kinds - {'text'})}). "
                    "Fireworks glm-5p2 is text-only -- keep vision calls on Anthropic.")
            content = "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
        out.append({"role": m.get("role", "user"), "content": content or ""})
    return out


class FireworksMessages:
    def __init__(self, api_key):
        self._key = api_key

    def create(self, model=None, max_tokens=4096, system=None, messages=None, **_ignored):
        """Same signature as Anthropic's messages.create, for the fields we use."""
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               _flatten(messages or [])
        payload = {
            "model": FIREWORKS_MODEL,
            "max_tokens": max_tokens,
            "messages": msgs,
            "reasoning_effort": REASONING_EFFORT,
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            FIREWORKS_URL, data=data,
            headers={"Authorization": f"Bearer {self._key}",
                     "Content-Type": "application/json",
                     # Cloudflare returns 1010 to urllib's default Python-urllib/3.x agent,
                     # which is how a Worker call once failed silently for an entire day.
                     "User-Agent": "shadow-gasp-pipeline"})

        last = None
        for attempt in range(3):
            try:
                # 600s was not enough for a long generation in CI while the same call
                # finished locally in minutes -- the timeout, not the model, was the limit.
                with urllib.request.urlopen(req, timeout=1800) as r:
                    body = json.load(r)
                break
            except urllib.error.HTTPError as e:
                detail = e.read()[:500].decode("utf-8", "replace")
                # 4xx other than 429 will not improve on retry: fail immediately with the
                # provider's own message rather than burning 3 attempts on a bad key.
                if 400 <= e.code < 500 and e.code != 429:
                    raise RuntimeError(f"Fireworks HTTP {e.code}: {detail}") from None
                last = f"HTTP {e.code}: {detail}"
            except Exception as e:  # noqa: BLE001 - network, timeout, malformed body
                last = repr(e)
            if attempt < 2:
                time.sleep(2 ** attempt * 3)
        else:
            raise RuntimeError(f"Fireworks call failed after 3 attempts: {last}")

        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        finish = choice.get("finish_reason")

        # Map the finish reason onto Anthropic's vocabulary so each caller's existing
        # truncation retry keeps working untouched. This is the whole point of the shim:
        # a reasoning model that spends its budget thinking and returns nothing is the
        # SAME recoverable condition as Anthropic's max_tokens, and must be reported as
        # such -- reported as anything else, the retry that would fix it never fires.
        stop_reason = "max_tokens" if finish == "length" else "end_turn"
        if not text.strip() and finish != "length":
            reasoning = message.get("reasoning_content") or ""
            hint = (f" It produced {len(reasoning)} chars of reasoning; raise max_tokens."
                    if reasoning else "")
            raise RuntimeError(
                f"Fireworks returned empty content (finish_reason={finish!r}).{hint}")
        return Response(text, stop_reason)


class FireworksClient:
    """Anthropic-shaped client backed by Fireworks."""

    def __init__(self, api_key):
        self.messages = FireworksMessages(api_key)


def provider():
    """Which provider this run will use. 'auto' resolves on which key is present."""
    choice = (os.environ.get("SHORT_LLM_PROVIDER") or "auto").strip().lower()
    if choice in ("anthropic", "fireworks"):
        return choice
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("FIREWORKS_API_KEY"):
        return "fireworks"
    raise RuntimeError(
        "No LLM key found. Set FIREWORKS_API_KEY (or ANTHROPIC_API_KEY), or pin one with "
        "SHORT_LLM_PROVIDER.")


def client():
    """Return a client exposing .messages.create(), whichever provider is configured."""
    name = provider()
    if name == "anthropic":
        from anthropic import Anthropic
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("SHORT_LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")
        print("LLM provider: anthropic", flush=True)
        return Anthropic(api_key=key)

    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        raise RuntimeError("SHORT_LLM_PROVIDER=fireworks but FIREWORKS_API_KEY is not set")
    print(f"LLM provider: fireworks ({FIREWORKS_MODEL})", flush=True)
    return FireworksClient(key)
