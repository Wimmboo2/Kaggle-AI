"""
Reasoning-stripping proxy for llama-cpp-python's OpenAI server.

Why this exists
---------------
The Qwen3 chat template baked into the GGUF ends the generation prompt with
a literal "<think>\n". The model therefore starts generating *inside* the
reasoning block and only ever emits the CLOSING "</think>" tag.

llama-cpp-python (0.3.x) has no reasoning parser at all -- it hands back the
raw completion -- so the OpenAI response looks like:

    choices[0].message.content == "okay so the user wants...</think>\n\nHere is the answer."

i.e. the chain of thought leaks into `content`, with no opening tag to match on.

This proxy sits in front of the llama.cpp server, splits on the first
"</think>", and puts the reasoning in `reasoning_content` (DeepSeek/Qwen
convention that OpenWebUI, SillyTavern, Cherry Studio, LibreChat, etc. render
as a collapsible "thinking" panel) while `content` holds only the answer.

Modes (env REASONING_MODE, or per-request "reasoning_mode" in the JSON body):
    separate  (default) -> reasoning moved to `reasoning_content`
    drop                -> reasoning discarded entirely
    raw                 -> passthrough, no rewriting (for debugging)
"""

import json
import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:8000").rstrip("/")
DEFAULT_MODE = os.environ.get("REASONING_MODE", "separate")
# Stream the chain of thought token-by-token as it is produced. Off by default:
# see StreamSplitter.feed for why holding it back is safer.
STREAM_REASONING = os.environ.get("STREAM_REASONING", "0") not in ("0", "", "false")
TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=None, pool=None)

OPEN, CLOSE = "<think>", "</think>"

app = FastAPI(title="think-stripping proxy")


def split_reasoning(text: str):
    """Return (reasoning, answer, found_close_tag)."""
    idx = text.find(CLOSE)
    if idx == -1:
        # No closing tag: generation was cut off mid-thought (max_tokens too
        # low), or thinking was disabled. Leave the text alone rather than
        # returning an empty message.
        return "", text, False
    head, tail = text[:idx], text[idx + len(CLOSE):]
    prefix = ""
    o = head.find(OPEN)
    if o != -1:  # model re-emitted the opening tag
        prefix, head = head[:o], head[o + len(OPEN):]
    return head.strip("\n"), (prefix + tail).lstrip("\n"), True


class StreamSplitter:
    """Incremental version of split_reasoning for SSE deltas."""

    def __init__(self, stream_reasoning: bool = False):
        self.stream_reasoning = stream_reasoning
        self.held = ""
        self.buf = ""
        self.in_reasoning = True
        self.saw_open = False
        self.pending_lstrip = True
        self.reasoning_so_far = ""

    def feed(self, piece: str):
        """Return (reasoning_delta, content_delta).

        By default the reasoning is held back until "</think>" actually
        arrives, then released in one delta. That way a stream that ends
        mid-thought (truncated, or thinking disabled) can be flushed as
        ordinary content instead of being mislabelled as reasoning."""
        was_reasoning = self.in_reasoning
        reasoning, content = self._feed(piece)
        if self.stream_reasoning:
            return reasoning, content
        if was_reasoning and self.in_reasoning:
            self.held += reasoning
            return "", content
        if was_reasoning:  # just closed
            reasoning, self.held = self.held + reasoning, ""
        return reasoning, content

    def _feed(self, piece: str):
        if not self.in_reasoning:
            return "", self._lstrip_once(piece)

        self.buf += piece

        # Decide once whether the reasoning opens with a literal "<think>".
        # The tag may be split across SSE chunks, so wait until there are
        # enough characters to tell.
        if not self.saw_open:
            lead = self.buf.lstrip()
            if lead.startswith(OPEN):
                self.buf = lead[len(OPEN):].lstrip("\n")
                self.saw_open = True
            elif len(lead) < len(OPEN) and OPEN.startswith(lead):
                return "", ""  # still ambiguous
            else:
                self.saw_open = True

        idx = self.buf.find(CLOSE)
        if idx != -1:
            head = self._lead(self.buf[:idx].rstrip("\n"))
            tail = self.buf[idx + len(CLOSE):]
            self.in_reasoning = False
            self.buf = ""
            self.reasoning_so_far += head
            return head, self._lstrip_once(tail)

        # Hold back any suffix that could be the start of a split "</think>",
        # and any whitespace right before it, so `reasoning_content` does not
        # end with the newline that precedes the closing tag.
        keep = 0
        for k in range(min(len(CLOSE) - 1, len(self.buf)), 0, -1):
            if self.buf.endswith(CLOSE[:k]):
                keep = k
                break
        cut = len(self.buf) - keep
        cut -= len(self.buf[:cut]) - len(self.buf[:cut].rstrip())
        emit, self.buf = self._lead(self.buf[:cut]), self.buf[cut:]
        self.reasoning_so_far += emit
        return emit, ""

    def _lead(self, s: str) -> str:
        """Drop the newline(s) that follow an opening <think> tag."""
        return s.lstrip("\n") if not self.reasoning_so_far else s

    def flush(self) -> str:
        """Called at end of stream. If "</think>" never arrived, thinking was
        either off or the generation was truncated mid-thought -- return the
        whole text so the client is never left with an empty message."""
        if not self.in_reasoning:
            return ""
        text = self.reasoning_so_far + self.buf
        self.in_reasoning = False
        self.buf = self.held = ""
        return text

    def _lstrip_once(self, s: str) -> str:
        """Swallow the blank line the template puts after </think>."""
        if not self.pending_lstrip:
            return s
        s = s.lstrip("\n")
        if s:
            self.pending_lstrip = False
        return s


def _apply(msg: dict, mode: str) -> None:
    content = msg.get("content")
    if not isinstance(content, str):
        return
    reasoning, answer, found = split_reasoning(content)
    if not found:
        return
    msg["content"] = answer
    if mode == "separate" and reasoning:
        msg["reasoning_content"] = reasoning


def _hop_headers(req: Request) -> dict:
    drop = {"host", "content-length", "connection", "accept-encoding"}
    return {k: v for k, v in req.headers.items() if k.lower() not in drop}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    mode = body.pop("reasoning_mode", DEFAULT_MODE)
    headers = _hop_headers(request)
    url = f"{UPSTREAM}/v1/chat/completions"

    if mode == "raw":
        return await _passthrough(request, "/v1/chat/completions", body)

    if not body.get("stream"):
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.post(url, json=body, headers=headers)
        if r.status_code != 200:
            return Response(r.content, status_code=r.status_code,
                            media_type=r.headers.get("content-type"))
        data = r.json()
        for ch in data.get("choices", []):
            if isinstance(ch.get("message"), dict):
                _apply(ch["message"], mode)
        return JSONResponse(data)

    # Connect eagerly so an upstream error (bad API key, model still loading)
    # comes back with its real status instead of a 200 with an empty stream.
    client = httpx.AsyncClient(timeout=TIMEOUT)
    upstream = await client.send(
        client.build_request("POST", url, json=body, headers=headers), stream=True
    )
    if upstream.status_code != 200:
        payload = await upstream.aread()
        await upstream.aclose()
        await client.aclose()
        return Response(payload, status_code=upstream.status_code,
                        media_type=upstream.headers.get("content-type"))

    async def gen():
        splitters = {}
        last_id, last_created, last_model = "chatcmpl-proxy", 0, body.get("model")
        try:
            async for line in upstream.aiter_lines():
                if not line.startswith("data: "):
                    if line:
                        yield f"{line}\n"
                    continue

                payload = line[6:].strip()
                if payload == "[DONE]":
                    # Nothing ever closed the reasoning block: emit what we held
                    # back as ordinary content so the reply is not empty.
                    for idx, sp in splitters.items():
                        leftover = sp.flush()
                        if leftover:
                            yield "data: " + json.dumps({
                                "id": last_id,
                                "object": "chat.completion.chunk",
                                "created": last_created,
                                "model": last_model,
                                "choices": [{"index": idx,
                                             "delta": {"content": leftover},
                                             "finish_reason": None}],
                            }) + "\n\n"
                    yield "data: [DONE]\n\n"
                    continue

                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    yield f"{line}\n\n"
                    continue

                last_id = chunk.get("id", last_id)
                last_created = chunk.get("created", last_created)
                last_model = chunk.get("model", last_model)

                for ch in chunk.get("choices", []):
                    delta = ch.get("delta")
                    if not isinstance(delta, dict):
                        continue
                    piece = delta.get("content")
                    if not isinstance(piece, str) or piece == "":
                        continue
                    sp = splitters.setdefault(
                        ch.get("index", 0), StreamSplitter(STREAM_REASONING))
                    reasoning, content = sp.feed(piece)
                    delta["content"] = content or None
                    if mode == "separate" and reasoning:
                        delta["reasoning_content"] = reasoning

                # Always forward the chunk, even when it is now empty: it keeps
                # the SSE connection warm while the model is still thinking.
                yield f"data: {json.dumps(chunk)}\n\n"
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


async def _passthrough(request: Request, path: str, body=None):
    headers = _hop_headers(request)
    url = f"{UPSTREAM}{path}"
    content = json.dumps(body).encode() if body is not None else await request.body()
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.request(request.method, url, content=content,
                                 headers=headers, params=request.query_params)
    return Response(r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type"))


@app.api_route("/{path:path}",
               methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def catch_all(request: Request, path: str):
    return await _passthrough(request, f"/{path}")
