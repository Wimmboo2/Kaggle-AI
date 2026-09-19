"""
Reasoning-stripping proxy for OpenAI-compatible llama.cpp servers.

Why this exists
---------------
Local runners (llama-cpp-python 0.3.x, older llama-server builds, anything
without a reasoning parser for the model you loaded) hand back the raw
completion, so the chain of thought ends up in `choices[].message.content`
and every frontend renders it as part of the reply.

It leaks in several different shapes:

  * Qwen3 / DeepSeek-R1 / QwQ / GLM / Phi-4-reasoning / Nemotron and the
    Gemma "thinking" finetunes wrap it in <think>...</think> -- and because
    the chat template ends the prompt with a literal "<think>\n", the model
    often emits ONLY the closing tag.
  * Magistral uses [THINK], Cohere uses <|START_THINKING|>, Kimi uses
    the fullwidth triangle variant, Seed-OSS uses <seed:think>, EXAONE Deep
    uses <thought>, the Unsloth-style Gemma reasoning tunes use
    <start_working_out>.
  * gpt-oss uses no tags at all. It uses harmony channels:
        <|channel|>analysis<|message|>...<|end|>
        <|start|>assistant<|channel|>final<|message|>the actual reply
    Only the `final` channel is meant to be shown.

This proxy sits in front of the upstream server and filters all of those out
of `content`, streaming and non-streaming, for every block in the message --
not just the first one.

Modes (env REASONING_MODE, or per-request "reasoning_mode" in the JSON body):
    separate  (default) -> reasoning moved to `reasoning_content`
    drop                -> reasoning discarded entirely
    raw                 -> passthrough, no rewriting (for debugging)

Other env knobs:
    UPSTREAM          base url of the real server (default http://127.0.0.1:8000)
    PREFILL_THINK     auto (default) | yes | no
                      How to treat text before any marker is seen.
                      auto: hold it back; a closing tag with no opener means
                            it was a prefilled thought, otherwise it is content.
                      yes:  the template definitely prefills "<think>" (Qwen).
                      no:   never assume a prefill -- lowest latency, streams
                            from the first token, only strips explicit blocks.
    ON_UNCLOSED       drop (default) | keep
                      What to do when the stream ends inside a thought
                      (max_tokens too low). `drop` means a truncated reply
                      comes back empty instead of leaking -- raise max_tokens.
    STREAM_REASONING  1 (default) -> reasoning_content streams live
                      0            -> held and sent in one delta at </think>
    SCRUB_HISTORY     1 (default) -> strip leaked thoughts out of the assistant
                      turns the client sends back up (Janitor keeps them)
    CORS_ORIGINS      * (default) -- Janitor AI calls this from the browser

Janitor AI notes
----------------
Janitor is a browser app, so it needs CORS headers (added below) and it will
happily be pointed at a url without the /v1 prefix, so both /chat/completions
and /v1/chat/completions are handled. Janitor does not render
`reasoning_content`, so `separate` and `drop` look identical there.
"""

import json
import os
import re
import sys

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:8000").rstrip("/")
DEFAULT_MODE = os.environ.get("REASONING_MODE", "separate")
PREFILL_THINK = os.environ.get("PREFILL_THINK", "auto").lower()
ON_UNCLOSED = os.environ.get("ON_UNCLOSED", "drop").lower()
SCRUB_HISTORY = os.environ.get("SCRUB_HISTORY", "1") not in ("0", "", "false")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")
# Truncated thoughts can only be recovered if the reasoning was withheld, so
# ON_UNCLOSED=keep forces the non-streaming variant.
STREAM_REASONING = (os.environ.get("STREAM_REASONING", "1") not in ("0", "", "false")
                    and ON_UNCLOSED != "keep")
WARN_UNKNOWN = os.environ.get("WARN_UNKNOWN", "1") not in ("0", "", "false")
TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=None, pool=None)

# ---------------------------------------------------------------------------
# Markers
#
# kind:
#   "hide"        start of a thought. Text seen before it (while undecided)
#                 was real content.
#   "show"        end of a thought / start of the visible answer. Text seen
#                 before it (while undecided) was a prefilled thought.
#   "hide_close"  ends a hidden region AND starts another one. Harmony's
#                 <|end|> / <|start|> sit between channels, so anything
#                 before them was reasoning, and what follows is still not
#                 the answer until a `final` channel opens.
#   "drop"        strip the marker, do not change state.
# ---------------------------------------------------------------------------
MARKERS = [
    # Qwen3, DeepSeek-R1, QwQ, GLM-4.5/4.6, Phi-4-reasoning, Nemotron,
    # Gemma thinking finetunes, and most community reasoning tunes.
    ("<think>", "hide"), ("</think>", "show"),
    ("<thinking>", "hide"), ("</thinking>", "show"),
    ("<thought>", "hide"), ("</thought>", "show"),            # EXAONE Deep
    ("<reasoning>", "hide"), ("</reasoning>", "show"),
    ("<reflection>", "hide"), ("</reflection>", "show"),
    # Mistral / Magistral
    ("[THINK]", "hide"), ("[/THINK]", "show"),
    # Cohere Command A / Command R7B
    ("<|START_THINKING|>", "hide"), ("<|END_THINKING|>", "show"),
    ("<|START_RESPONSE|>", "drop"), ("<|END_RESPONSE|>", "drop"),
    # Moonshot Kimi (fullwidth triangles, not ASCII angle brackets)
    ("◁think▷", "hide"), ("◁/think▷", "show"),
    # ByteDance Seed-OSS
    ("<seed:think>", "hide"), ("</seed:think>", "show"),
    # Unsloth / Gemma-style reasoning finetunes
    ("<start_working_out>", "hide"), ("<end_working_out>", "show"),
    ("<SOLUTION>", "drop"), ("</SOLUTION>", "drop"),
    # Sky-T1 / OpenThoughts
    ("<|begin_of_thought|>", "hide"), ("<|end_of_thought|>", "show"),
    ("<|begin_of_solution|>", "drop"), ("<|end_of_solution|>", "drop"),
    # Mirrored-pipe channel variant, observed in the wild:
    #     <|channel>thought ...reasoning... <channel|>the answer
    # The pipe sits inside-left to open and inside-right to close, so unlike
    # canonical harmony below these ARE a plain pair and the name after the
    # opener ("thought") is not a routing decision. Matching only the
    # both-pipes "<|channel|>" form misses this entirely.
    ("<|channel>", "hide"), ("<channel|>", "show"),
    ("<|think>", "hide"), ("<think|>", "show"),
    ("<|thinking>", "hide"), ("<thinking|>", "show"),
    ("<|thought>", "hide"), ("<thought|>", "show"),
    ("<|reasoning>", "hide"), ("<reasoning|>", "show"),
    # gpt-oss / harmony proper. `final` is the only channel the user should
    # see; the bare <|channel|> catches analysis and commentary.
    ("<|channel|>", "hide"),
    ("<|message|>", "drop"),
    ("<|constrain|>", "drop"),
    ("<|end|>", "hide_close"),
    ("<|start|>", "hide_close"),
    ("<|call|>", "hide_close"),
    ("<|return|>", "hide_close"),
]

_KIND = {lit.lower(): kind for lit, kind in MARKERS}
# Harmony writes a short header between <|channel|> / <|start|> and
# <|message|> -- "analysis", "assistant", "final", "commentary to=functions.f".
# It is routing metadata rather than thought, and it is also the ONLY thing
# that says whether the message body about to start is the answer or not:
#
#   <|start|>assistant<|channel|>analysis<|message|> thought  <|end|>
#   <|start|>assistant<|channel|>final<|message|>    answer   <|return|>
#
# <|channel|> appears in both, so it cannot decide anything on its own. The
# header is captured and read at <|message|>: `final` means show, anything
# else means keep hiding. Matching the whole "<|channel|>final<|message|>"
# run as one literal would break on any spacing variant, and breaking that
# way hides the entire answer.
_HEADER_OPEN = {"<|channel|>", "<|start|>"}
_HEADER_KEEP = {"<|constrain|>"}
_FINAL_CHANNEL = "final"
# Longest alternative first so the leftmost match is also the longest one
# (<|channel|>final<|message|> must win over a bare <|channel|>).
_PATTERN = re.compile(
    "|".join(re.escape(lit) for lit, _ in sorted(MARKERS, key=lambda m: -len(m[0]))),
    re.IGNORECASE,
)
# Every proper prefix of every marker, for holding back a marker that got
# split across two SSE chunks.
_PREFIXES = {lit[:i].lower() for lit, _ in MARKERS for i in range(1, len(lit))}
_MAXLEN = max(len(lit) for lit, _ in MARKERS)


class ThinkFilter:
    """Incremental chain-of-thought stripper.

    feed() takes any slice of the completion and returns
    (reasoning_delta, content_delta); flush() finishes the message.
    """

    UNDECIDED, HIDDEN, VISIBLE = 0, 1, 2

    def __init__(self, start=None, stream_reasoning=True, on_unclosed="drop"):
        if start is None:
            start = {"yes": self.HIDDEN, "no": self.VISIBLE}.get(
                PREFILL_THINK, self.UNDECIDED)
        self.state = start
        self.stream_reasoning = stream_reasoning
        self.on_unclosed = on_unclosed
        self.buf = ""       # unprocessed tail (may hold a partial marker)
        self.held = ""      # text buffered while UNDECIDED
        self.pending = ""   # reasoning withheld when stream_reasoning is off
        self.r_out = []
        self.c_out = []
        self.lstrip_reason = True
        self.lstrip_content = True
        self.saw_content = False
        self.in_header = False
        self.header = None
        self.saw_marker = False

    # -- emit helpers -------------------------------------------------------
    def _reason(self, s):
        if not s:
            return
        if self.lstrip_reason:
            s = s.lstrip("\n")
            if not s:
                return
            self.lstrip_reason = False
        if self.stream_reasoning:
            self.r_out.append(s)
        else:
            self.pending += s

    def _content(self, s):
        if not s:
            return
        if self.lstrip_content:
            s = s.lstrip("\n")
            if not s:
                return
            self.lstrip_content = False
        self.saw_content = True
        self.c_out.append(s)

    def _text(self, s):
        if not s:
            return
        if self.in_header:
            self.header = (self.header or "") + s
            return
        if self.state == self.UNDECIDED:
            self.held += s
        elif self.state == self.HIDDEN:
            self._reason(s)
        else:
            self._content(s)

    # -- state transitions --------------------------------------------------
    def _go_hidden(self, held_was_reasoning):
        if self.state == self.UNDECIDED:
            head, self.held = self.held, ""
            if held_was_reasoning:
                self.state = self.HIDDEN
                self._reason(head)
            else:
                self.state = self.VISIBLE
                self._content(head)
        self.state = self.HIDDEN
        self.lstrip_reason = True

    def _go_visible(self):
        if self.state == self.UNDECIDED:
            # A closing tag with nothing opening it: the template prefilled
            # the opener, so everything so far was the thought.
            head, self.held = self.held, ""
            self.state = self.HIDDEN
            self._reason(head)
        if self.pending:
            self.r_out.append(self.pending)
            self.pending = ""
        self.state = self.VISIBLE
        self.lstrip_content = True

    # -- driving ------------------------------------------------------------
    def _drain(self, final):
        while True:
            m = _PATTERN.search(self.buf)
            if not m:
                break
            rest = self.buf[m.start():]
            if not final and len(rest) < _MAXLEN and rest.lower() in _PREFIXES:
                # A complete marker that is also the start of a longer one:
                # "<|channel|>" may still turn into "<|channel|>final<|message|>",
                # which means the opposite thing. Wait for more tokens.
                break
            self._text(self.buf[:m.start()])
            self.buf = self.buf[m.end():]
            lit = m.group(0).lower()
            kind = _KIND[lit]
            self.saw_marker = True
            if lit == "<|message|>" and self.header is not None:
                # End of a harmony header: the name decides what follows.
                name, self.header = self.header.strip().lower(), None
                self.in_header = False
                if name.startswith(_FINAL_CHANNEL):
                    self._go_visible()
                else:
                    self._go_hidden(True)
                continue
            if lit not in _HEADER_KEEP:
                self.in_header = lit in _HEADER_OPEN
                self.header = "" if self.in_header else None
            if kind == "hide":
                self._go_hidden(False)
            elif kind == "hide_close":
                self._go_hidden(True)
            elif kind == "show":
                self._go_visible()
            # "drop": marker removed, state untouched

        if final:
            self._text(self.buf)
            self.buf = ""
            return
        # Hold back a suffix that could be the front of a split marker, plus
        # the whitespace in front of it, so a thought does not end with the
        # newline that precedes its closing tag.
        cut = len(self.buf) - self._partial_len(self.buf)
        while cut > 0 and self.buf[cut - 1] in " \t\r\n":
            cut -= 1
        self._text(self.buf[:cut])
        self.buf = self.buf[cut:]

    def feed(self, piece):
        self.buf += piece
        self._drain(final=False)
        return self._take()

    def flush(self, truncated=False):
        """End of message. `truncated` is finish_reason == "length", i.e. the
        model hit max_tokens. Returns the final (reasoning, content) deltas."""
        self.in_header = False
        self.header = None
        self._drain(final=True)
        if self.state == self.UNDECIDED:
            head, self.held = self.held, ""
            if truncated and not self.saw_marker and PREFILL_THINK == "auto":
                # Cut off by max_tokens with no marker anywhere in the output.
                # A template that ends the prompt with a literal "<think>" makes
                # exactly this shape: the model is still inside the thought it
                # was handed, so there is no opening tag to find and no closing
                # tag was ever reached. Textually identical to a model that just
                # did not think -- finish_reason is the only thing that tells
                # them apart. Treat it as a thought rather than print it.
                self.state = self.HIDDEN
                self._reason(head)
                self._end_unclosed()
            else:
                self.state = self.VISIBLE
                self._content(head)
        elif self.state == self.HIDDEN:
            # Stopped inside an explicitly opened thought. Never fall back to
            # printing it -- that was the old behaviour and it is the leak.
            self._end_unclosed()
        return self._take()

    def _end_unclosed(self):
        if self.pending:
            if self.on_unclosed == "keep":
                self.state = self.VISIBLE
                self.lstrip_content = True
                self._content(self.pending)
            else:
                # Still reasoning, just unfinished -- surface it in
                # reasoning_content rather than discarding it silently, so the
                # non-streaming path matches the streaming one.
                self.r_out.append(self.pending)
        self.pending = ""
        self.state = self.VISIBLE

    def _take(self):
        r, c = "".join(self.r_out), "".join(self.c_out)
        self.r_out, self.c_out = [], []
        return r, c

    @staticmethod
    def _partial_len(buf):
        low = buf.lower()
        n = len(low)
        for k in range(min(_MAXLEN - 1, n), 0, -1):
            if low[n - k:] in _PREFIXES:
                return k
        return 0


# Anything shaped like a special token: <|x|>, <|x>, <x|>. If one of these
# reaches the client it is a marker this file does not know about, which is
# how a leak starts. Name it on stderr so it can be added to MARKERS instead
# of being rediscovered from a chat log.
_SUSPECT = re.compile(r"<\|[^<>|\s]{1,24}\|?>|<[^<>|\s]{1,24}\|>")
_WARNED = set()


def warn_unknown(text):
    if not (WARN_UNKNOWN and text):
        return
    for tok in _SUSPECT.findall(text):
        if tok.lower() in _KIND or tok in _WARNED:
            continue
        _WARNED.add(tok)
        print(f"[thinkproxy] unrecognised marker reached the client: {tok!r} "
              f"-- add it to MARKERS", file=sys.stderr, flush=True)


def strip_reasoning(text, truncated=False, **kw):
    """Whole-string version. Returns (reasoning, content)."""
    f = ThinkFilter(**kw)
    r1, c1 = f.feed(text)
    r2, c2 = f.flush(truncated)
    return r1 + r2, c1 + c2


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
app = FastAPI(title="think-stripping proxy")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS.split(",")] if CORS_ORIGINS != "*" else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

_DROP_HEADERS = {"host", "content-length", "connection", "accept-encoding",
                 "origin", "referer", "cookie"}


def _hop_headers(req):
    return {k: v for k, v in req.headers.items()
            if k.lower() not in _DROP_HEADERS and not k.lower().startswith("sec-")}


def _new_filter():
    return ThinkFilter(stream_reasoning=STREAM_REASONING, on_unclosed=ON_UNCLOSED)


def _apply(obj, key, mode, truncated=False):
    """Filter obj[key] in place (message.content or completion text)."""
    text = obj.get(key)
    if not isinstance(text, str) or not text:
        return
    reasoning, answer = strip_reasoning(
        text, truncated=truncated, stream_reasoning=False,
        on_unclosed=ON_UNCLOSED)
    warn_unknown(answer)
    obj[key] = answer
    if mode == "separate" and reasoning:
        obj["reasoning_content"] = (obj.get("reasoning_content") or "") + reasoning
    elif mode == "drop":
        obj.pop("reasoning_content", None)


def _scrub_request(body):
    """Janitor (and most frontends) replay the whole chat every turn. If a
    thought leaked once it comes straight back up in the history, so clean the
    assistant turns on the way in."""
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        msg.pop("reasoning_content", None)
        content = msg.get("content")
        if isinstance(content, str) and content:
            _, msg["content"] = strip_reasoning(content, stream_reasoning=False)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    _, part["text"] = strip_reasoning(
                        part["text"], stream_reasoning=False)


async def _handle_completions(request, path, chat):
    body = await request.json()
    mode = body.pop("reasoning_mode", DEFAULT_MODE)
    if mode == "raw":
        return await _passthrough(request, path, body)

    if SCRUB_HISTORY and chat:
        _scrub_request(body)

    headers = _hop_headers(request)
    url = f"{UPSTREAM}{path}"
    key = "message" if chat else None

    if not body.get("stream"):
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.post(url, json=body, headers=headers)
        if r.status_code != 200:
            return Response(r.content, status_code=r.status_code,
                            media_type=r.headers.get("content-type"))
        data = r.json()
        for ch in data.get("choices", []):
            cut = ch.get("finish_reason") == "length"
            if chat and isinstance(ch.get(key), dict):
                _apply(ch[key], "content", mode, cut)
            elif not chat:
                _apply(ch, "text", mode, cut)
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
        filters = {}
        cut_off = set()
        last = {"id": "chatcmpl-proxy", "created": 0, "model": body.get("model")}
        try:
            async for line in upstream.aiter_lines():
                if not line.startswith("data: "):
                    if line:
                        yield f"{line}\n"
                    continue

                payload = line[6:].strip()
                if payload == "[DONE]":
                    for idx, f in filters.items():
                        reasoning, content = f.flush(idx in cut_off)
                        if not (reasoning or content):
                            continue
                        delta = {}
                        if content:
                            delta["content"] = content
                        if reasoning and mode == "separate":
                            delta["reasoning_content"] = reasoning
                        if not delta:
                            continue
                        yield "data: " + json.dumps({
                            "id": last["id"],
                            "object": "chat.completion.chunk" if chat else "text_completion",
                            "created": last["created"],
                            "model": last["model"],
                            "choices": [({"index": idx, "delta": delta,
                                          "finish_reason": None} if chat else
                                         {"index": idx, "text": delta.get("content", ""),
                                          "finish_reason": None})],
                        }) + "\n\n"
                    yield "data: [DONE]\n\n"
                    continue

                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    yield f"{line}\n\n"
                    continue

                for k in last:
                    if chunk.get(k) is not None:
                        last[k] = chunk[k]

                for ch in chunk.get("choices", []):
                    if ch.get("finish_reason") == "length":
                        cut_off.add(ch.get("index", 0))
                    if chat:
                        delta = ch.get("delta")
                        if not isinstance(delta, dict):
                            continue
                        if mode == "drop":
                            delta.pop("reasoning_content", None)
                        piece = delta.get("content")
                        if not isinstance(piece, str) or piece == "":
                            continue
                        f = filters.setdefault(ch.get("index", 0), _new_filter())
                        reasoning, content = f.feed(piece)
                        # Send no key at all rather than null: naive clients
                        # concatenate the delta blindly and print "null".
                        if content:
                            warn_unknown(content)
                            delta["content"] = content
                        else:
                            delta.pop("content", None)
                        if reasoning and mode == "separate":
                            delta["reasoning_content"] = (
                                delta.get("reasoning_content") or "") + reasoning
                    else:
                        piece = ch.get("text")
                        if not isinstance(piece, str) or piece == "":
                            continue
                        f = filters.setdefault(ch.get("index", 0), _new_filter())
                        _, content = f.feed(piece)
                        ch["text"] = content

                # Always forward the chunk, even when it is now empty: it keeps
                # the SSE connection warm while the model is still thinking.
                yield f"data: {json.dumps(chunk)}\n\n"
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


async def _passthrough(request, path, body=None):
    headers = _hop_headers(request)
    url = f"{UPSTREAM}{path}"
    content = json.dumps(body).encode() if body is not None else await request.body()
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.request(request.method, url, content=content,
                                 headers=headers, params=request.query_params)
    return Response(r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type"))


@app.get("/healthz")
async def healthz():
    return {"ok": True, "upstream": UPSTREAM, "mode": DEFAULT_MODE,
            "prefill_think": PREFILL_THINK, "on_unclosed": ON_UNCLOSED}


@app.api_route("/{path:path}",
               methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def catch_all(request: Request, path: str):
    # Match on the suffix, not the exact path: frontends get pointed at
    # /v1/chat/completions, /chat/completions and /api/v1/chat/completions
    # depending on what the user pasted into the box. Anything that falls
    # through to a raw passthrough leaks the whole thought.
    clean = "/" + path.strip("/")
    if request.method == "POST":
        if clean.endswith("/chat/completions"):
            return await _handle_completions(request, clean, chat=True)
        if clean.endswith("/completions"):
            return await _handle_completions(request, clean, chat=False)
    return await _passthrough(request, clean if path else "/")
