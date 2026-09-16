#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vllm_client.py - vllm_kestrel OpenAI-compatible HTTP client (stdlib only).

对接引擎的 HTTP 接口（见 src/serve/vllm_server.c）:
    GET  /v1/models
    GET  /health
    POST /v1/chat/completions   (stream=true 时返回 SSE)
    POST /v1/completions        (legacy 文本补全)

零第三方依赖，Windows / RK3588 (Python 3.6+) 均可直接运行。

CLI 用法:
    python tools/client/vllm_client.py --url http://127.0.0.1:8080 health
    python tools/client/vllm_client.py --url http://127.0.0.1:8080 models
    python tools/client/vllm_client.py chat "你好" --stream --max-tokens 128
    python tools/client/vllm_client.py chat --system "你是一个助手" "介绍一下自己"
    python tools/client/vllm_client.py chat --image photo.jpg "这张图里有什么？" --stream
    python tools/client/vllm_client.py complete "The capital of France is"
    echo "你好" | python tools/client/vllm_client.py chat
"""

import argparse
import base64
import json
import mimetypes
import sys
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "http://127.0.0.1:8080"


class VLLMError(Exception):
    """服务端返回非 2xx 时抛出，携带错误详情。"""


class VLLMClient:
    def __init__(self, base_url=DEFAULT_BASE_URL, timeout=300.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    # ---------- 基础请求 ----------

    def _request(self, method, path, payload=None):
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base + path, data=data,
                                     headers=headers, method=method)
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            # 服务端错误体为 OpenAI 风格 {"error":{"message":...}}
            try:
                err = json.loads(e.read().decode("utf-8", "replace"))
                msg = err.get("error", {}).get("message", str(e))
            except Exception:
                msg = str(e)
            raise VLLMError("%s %s -> %d: %s" % (method, path, e.code, msg))
        except urllib.error.URLError as e:
            raise VLLMError("%s %s -> 连接失败: %s" % (method, path, e.reason))

    def _json(self, method, path, payload=None):
        with self._request(method, path, payload) as r:
            return json.loads(r.read().decode("utf-8"))

    # ---------- 元数据接口 ----------

    def health(self):
        """GET /health: {status, model, busy, queued, active, ...}"""
        return self._json("GET", "/health")

    def models(self):
        """GET /v1/models"""
        return self._json("GET", "/v1/models")

    # ---------- 推理接口 ----------

    def chat(self, messages, temperature=0.7, top_p=0.9, max_tokens=64,
             stream=False):
        """POST /v1/chat/completions. stream=True 返回 SSE 块列表并在
        生成过程中实时打印增量文本。"""
        payload = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if not stream:
            return self._json("POST", "/v1/chat/completions", payload)
        return self._chat_stream(payload)

    def _chat_stream(self, payload):
        chunks = []
        with self._request("POST", "/v1/chat/completions", payload) as r:
            for raw in r:   # SSE 帧按行读取: "data: {...}\n\n"
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                chunks.append(obj)
                delta = obj.get("choices", [{}])[0].get("delta", {})
                piece = delta.get("content")
                if piece:
                    print(piece, end="", flush=True)
        print()
        return chunks

    def complete(self, prompt, temperature=0.7, top_p=0.9, max_tokens=64):
        """POST /v1/completions (legacy 文本补全，非流式)。"""
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        return self._json("POST", "/v1/completions", payload)

    # ---------- 多模态辅助 ----------

    @staticmethod
    def data_url(path):
        """本地图片/帧文件 -> OpenAI image_url 所需的 base64 data URL。"""
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        return "data:%s;base64,%s" % (mime, b64)


# ---------------- CLI ----------------

def _result_text(obj, kind):
    """从非流式响应中取出可打印的文本。"""
    try:
        if kind == "chat":
            return obj["choices"][0]["message"]["content"]
        return obj["choices"][0]["text"]
    except (KeyError, IndexError):
        return json.dumps(obj, ensure_ascii=False)


def _build_messages(prompt, system, images, video_frames):
    """组装 messages。纯文本时 content 为字符串；含媒体时用 part 数组。"""
    content = []
    if prompt:
        content.append({"type": "text", "text": prompt})
    for path in (images or []):
        content.append({"type": "image_url",
                        "image_url": {"url": VLLMClient.data_url(path)}})
    for path in (video_frames or []):
        content.append({"type": "video_frames", "video_frames": [VLLMClient.data_url(path)]})
    if len(content) == 1 and content[0]["type"] == "text":
        content = content[0]["text"]

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    return messages


def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(
        prog="vllm_client",
        description="vllm_kestrel OpenAI-compatible HTTP client (stdlib only)")
    ap.add_argument("--url", default=DEFAULT_BASE_URL,
                    help="引擎地址，默认 %s" % DEFAULT_BASE_URL)
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="请求超时秒数，默认 300")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("health", help="GET /health")
    sub.add_parser("models", help="GET /v1/models")

    p_chat = sub.add_parser("chat", help="POST /v1/chat/completions")
    p_chat.add_argument("prompt", nargs="?", help="用户消息（缺省时读 stdin）")
    p_chat.add_argument("--system", help="system 消息")
    p_chat.add_argument("--image", action="append", default=None,
                        help="图片路径，可多次，自动转 base64")
    p_chat.add_argument("--video-frame", action="append", default=None,
                        help="视频帧路径，可多次（需同尺寸）")
    p_chat.add_argument("--stream", action="store_true", help="SSE 流式输出")
    p_chat.add_argument("--max-tokens", type=int, default=64)
    p_chat.add_argument("--temperature", type=float, default=0.7)
    p_chat.add_argument("--top-p", type=float, default=0.9)

    p_comp = sub.add_parser("complete", help="POST /v1/completions (legacy)")
    p_comp.add_argument("prompt", nargs="?", help="提示文本（缺省时读 stdin）")
    p_comp.add_argument("--max-tokens", type=int, default=64)
    p_comp.add_argument("--temperature", type=float, default=0.7)
    p_comp.add_argument("--top-p", type=float, default=0.9)

    args = ap.parse_args()
    client = VLLMClient(args.url, args.timeout)

    if args.cmd == "health":
        print(json.dumps(client.health(), ensure_ascii=False, indent=2))
    elif args.cmd == "models":
        print(json.dumps(client.models(), ensure_ascii=False, indent=2))
    elif args.cmd == "chat":
        prompt = args.prompt if args.prompt is not None else sys.stdin.read()
        messages = _build_messages(prompt, args.system,
                                   args.image, args.video_frame)
        if args.stream:
            client.chat(messages, args.temperature, args.top_p,
                        args.max_tokens, stream=True)
        else:
            obj = client.chat(messages, args.temperature, args.top_p,
                              args.max_tokens, stream=False)
            print(_result_text(obj, "chat"))
    elif args.cmd == "complete":
        prompt = args.prompt if args.prompt is not None else sys.stdin.read()
        obj = client.complete(prompt, args.temperature, args.top_p,
                              args.max_tokens)
        print(_result_text(obj, "complete"))


if __name__ == "__main__":
    main()
