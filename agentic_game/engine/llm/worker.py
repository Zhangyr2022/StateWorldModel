#!/usr/bin/env python3
"""常驻 LLM worker:transformers 加载大模型,分片驻留全部 GPU。

为什么存在:Qwen 27B 权重 ~52GB,逐请求装载不现实。worker 以
device_map="auto" 把模型按层切分到 8 张卡(每卡约 7GB,与常驻的
Minimax-H3 服务共存),之后每次对话只是本地 generate。

协议(Unix socket,JSON Lines,与 h3_worker 同风格):
- {"cmd": "ping"}                     → {"event": "pong", ...}
- {"cmd": "shutdown"}                 → {"event": "bye"} 后退出
- {"messages": [...], "seed": N, "max_tokens": M}
                                      → {"event": "chat", "ok": true, "text": ...}

解码用贪心(do_sample=False):同输入同输出,配合游戏层的 seed 约定
可以近似复现(模型服务有状态,严格确定性以 greedy 为上限)。

启动:python3 -m engine.llm.service start(或直接 torchrun 无关,单进程即可)。
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import sys
import time
from pathlib import Path

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think>.*\Z", flags=re.DOTALL)  # 被 max_tokens 截断的思考块


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--per-gpu-mem", default="7GiB",
                        help="每张卡允许的模型显存上限(与 H3 共存时留出余量;52GB 模型 8 卡,7GiB 是下限安全值)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


class LLMService:
    def __init__(self, model_path: str, per_gpu_mem: str, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoProcessor

        from transformers import AutoModelForImageTextToText

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        n_gpus = torch.cuda.device_count()
        max_memory = {i: per_gpu_mem for i in range(n_gpus)}
        max_memory["cpu"] = "0GiB"  # 装不下就明确失败,禁止静默 CPU offload 拖慢推理

        # Qwen3.5 是 VL 条件生成架构:processor 同时管文本模板与视觉预处理
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype="bfloat16",
            device_map="auto",
            max_memory=max_memory,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.renders = 0  # 已完成的对话数

    def chat(self, messages: list[dict], seed: int, max_tokens: int | None,
             images: list[str] | None = None) -> str:
        torch = self.torch
        vl_messages = messages
        pil_images = []
        if images:
            # VL 模式:图片并入最后一条 user 消息(先图后文)
            pil_images = [_load_image(path) for path in images]
            vl_messages = [dict(m) for m in messages]
            last_user = next(m for m in reversed(vl_messages) if m.get("role") == "user")
            content: list[dict] = [{"type": "image"} for _ in pil_images]
            content.append({"type": "text", "text": last_user.get("content", "")})
            last_user["content"] = content

        # 标准 VL 流程:模板产出带 <|image_pad|> 占位的文本串,
        # processor 再把真实图像特征填进去
        try:
            prompt_text = self.processor.apply_chat_template(
                vl_messages, add_generation_prompt=True, enable_thinking=False,
            )
        except (TypeError, ValueError):  # 模板不认识 enable_thinking
            prompt_text = self.processor.apply_chat_template(vl_messages, add_generation_prompt=True)

        if pil_images:
            inputs = self.processor(text=[prompt_text], images=pil_images, return_tensors="pt")
        else:
            inputs = self.processor(text=[prompt_text], return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        # transformers 5.8 的 generate() 不接受 generator kwarg;
        # 贪心解码本身与随机数无关,这里设全局种子只为未来切换采样时可用。
        torch.manual_seed(int(seed))
        started = time.monotonic()
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens or self.max_new_tokens,
                do_sample=False,
            )
        elapsed = time.monotonic() - started
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output[0][prompt_len:]
        text = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        # 模板不支持关闭思考时,输出会带 <think>...</think>(或被截断的未闭合
        # 思考块);统一剥掉,剥完为空视为一次失败输出。
        text = _THINK_RE.sub("", _THINK_UNCLOSED_RE.sub("", text)).strip()
        return text, elapsed


def _load_image(path: str):
    """加载并适度缩小图片(VL 编码成本随分辨率平方增长,896px 足够看内容)。"""
    from PIL import Image

    image = Image.open(path).convert("RGB")
    long_side = max(image.size)
    if long_side > 896:
        scale = 896 / long_side
        image = image.resize((int(image.width * scale), int(image.height * scale)))
    return image


def serve(socket_path: str, service: LLMService) -> None:
    path = Path(socket_path)
    path.unlink(missing_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(1)
    print(json.dumps({"event": "listening", "socket": str(path)}), flush=True)

    shutting_down = False
    try:
        while not shutting_down:
            conn, _ = server.accept()
            with conn:
                file = conn.makefile("r")
                while True:
                    line = file.readline()
                    if not line:
                        break  # 客户端断开,等下一个连接
                    try:
                        request = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(request, dict):
                        continue
                    if request.get("cmd") == "ping":
                        reply = {"event": "pong", "calls": service.renders,
                                 "device": str(service.device)}
                    elif request.get("cmd") == "shutdown":
                        reply = {"event": "bye"}
                        shutting_down = True
                    elif request.get("messages"):
                        result: dict = {"event": "chat"}
                        try:
                            text, elapsed = service.chat(
                                request["messages"],
                                seed=int(request.get("seed", 0)),
                                max_tokens=request.get("max_tokens"),
                                images=request.get("images"),
                            )
                            result.update(ok=True, text=text, elapsed_s=round(elapsed, 2))
                            service.renders += 1
                        except Exception as exc:  # noqa: BLE001 - 错误原样上报给调用方
                            result.update(ok=False, error_type=type(exc).__name__,
                                          message=str(exc)[:2000])
                        reply = result
                    else:
                        continue
                    try:
                        conn.sendall((json.dumps(reply, ensure_ascii=False)
                                      + "\n").encode("utf-8"))
                    except OSError as exc:
                        # 客户端中途断连(评测任务被 kill 时易撞上回包)只断开
                        # 当前连接;worker 常驻,继续服务下一个连接。
                        print(json.dumps({"event": "client_disconnected",
                                          "error": str(exc)[:200]}), flush=True)
                        break
                    if shutting_down:
                        break
    finally:
        server.close()
        path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    service = LLMService(args.model, args.per_gpu_mem, args.max_new_tokens)
    serve(args.socket, service)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
