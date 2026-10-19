#!/usr/bin/env python3
"""离线 LLM 桩服务:与常驻 LLM 服务(engine.llm.service)同一 socket 协议,
但不装载任何模型——纯 CPU,让 LLM 参与结算的游戏(romance / adv)在
fake renderer 下做整局冒烟,不需要 GPU 与权重。

应答规则(确定性;按"对话内容哈希"从台词库选一条,同一输入同一回应,
不同输入不同回应——避免固定周期复读):
- system 提示含"【角色】"(adv 女主)→ 返回合法 JSON 契约的中文回应;
  其中 system 含"本回合刚发生"(剧情节点开场,node_event)时改用
  节点事件专属回应,让女主对刚发生的固定事件有反应;
- 其余(LLM 玩家代理等)→ 返回一句中文台词。

用法:
    python3 scripts/dev_llm_stub.py --socket /tmp/agentic-game-llm-stub.sock
    # 另开终端:
    python3 run_demo.py --game before_rain_mio --renderer fake --agent auto \
        --llm-socket /tmp/agentic-game-llm-stub.sock

这只是开发/测试装置:应答与真实模型无关,不能用于任何体验或数值评估。
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import zlib
from pathlib import Path

# 桩女主的回应轮换(全部在合法 JSON 契约内;关系值有正有负;中文台词)。
# 选择按"对话内容哈希"而不是调用计数:玩家说不同的话,回应就不同,
# 不再出现固定周期复读;同一输入仍给同一回应(确定性)。
_GIRL_REPLIES = [
    {"reply": "又是你啊。……好吧,那摞照片递给我。", "action": "tucking a loose strand of hair behind her ear, eyes flicking to you for a beat",
     "relationship_delta": 1, "mood": "calm"},
    {"reply": "这话……还算中听。别得意。", "action": "lips pressing into a small pout, cheeks warming as she looks away",
     "relationship_delta": 2, "mood": "warm"},
    {"reply": "嗯。我想先把这一页做完。……不过,谢谢。", "action": "glancing up briefly through her lashes, fingers pausing on the page",
     "relationship_delta": 1, "mood": "guarded"},
    {"reply": "这么晚了还净说些奇怪的话。快点回家吧。", "action": "rubbing one eye with a knuckle, shoulders slumping a little",
     "relationship_delta": 0, "mood": "tired"},
    {"reply": "……没什么。只是觉得这张照片,拍得有点像你。", "action": "hugging the photo print to her chest, ears turning pink",
     "relationship_delta": 2, "mood": "flustered"},
    {"reply": "随便你怎么说。反正纪念册总要做完的。",
     "relationship_delta": 0, "mood": "calm"},
    {"reply": "你总是这样,哪句话是真的,哪句话是客气?",
     "relationship_delta": 1, "mood": "guarded"},
    {"reply": "……就这一次。下不为例。",
     "relationship_delta": 2, "mood": "warm"},
]
# 节点开场(node_event,system 出现"本回合刚发生")的专属回应:
# 按【当前剧情】文本里的关键词匹配节点——女主对"刚发生的固定事件"作出
# 与该事件对应的反应(下雨→被困、工作被做完、旧照片、纪念册完成),而非接话。
_NODE_EVENT_REPLIES = [
    ("大雨",
     {"reply": "……外面下起大雨了呢。今天,大概谁都走不了吧。",
      "relationship_delta": 1, "mood": "calm"}),
    ("工作已经全部",
     {"reply": "这些页……我做完了。昨天刚好有时间而已。",
      "relationship_delta": 0, "mood": "tired"}),
    ("搬家",
     {"reply": "咳……没什么。只是旧照片而已,不要那样看着我。",
      "relationship_delta": 0, "mood": "sad"}),
    ("纪念册完成了",
     {"reply": "完成了。明天开始……你不用来了。",
      "relationship_delta": -1, "mood": "guarded"}),
]
_PLAYER_LINES = [
    "今天雨下得真大,谁都走不了了。",
    "你昨天看起来很累,昨晚没睡吗?",
    "这些旧照片里的你,都在笑呢。",
    "就算毕业以后,今天的时光也作数吧。",
    "纪念册做完之后,我明天还能来吗?",
]


def _serve_conn(conn: socket.socket, state: dict) -> None:
    buffer = b""
    try:
        while True:
            while b"\n" not in buffer:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                buffer += chunk
            line, buffer = buffer.split(b"\n", 1)
            if not line.strip():
                continue
            request = json.loads(line.decode("utf-8"))
            conn.sendall((json.dumps(_respond(request, state),
                                     ensure_ascii=False) + "\n").encode("utf-8"))
    except (json.JSONDecodeError, OSError):
        pass  # 坏请求/断连:关闭本连接即可,服务继续
    finally:
        conn.close()


def _pick(bank: list, key: str) -> object:
    """按内容哈希从台词库取一条:同一输入同一回应,不同输入不同回应。"""
    return bank[zlib.crc32(key.encode("utf-8")) % len(bank)]


def _last_user(messages: list[dict]) -> str:
    return next((m.get("content", "") for m in reversed(messages)
                 if m.get("role") == "user"), "")


def _respond(request: dict, state: dict) -> dict:
    if request.get("cmd") == "ping":
        return {"event": "pong", "calls": state["calls"], "stub": True}
    state["calls"] += 1
    messages = request.get("messages") or []
    system = next((m.get("content", "") for m in messages
                   if m.get("role") == "system"), "")
    key = f"{request.get('seed', 0)}:{_last_user(messages)}"
    if "【角色】" in system:
        if "本回合刚发生" in system:
            # 节点开场:按剧情文本关键词取该节点的事件回应(兜底走内容哈希)
            reply = next((item for keyword, item in _NODE_EVENT_REPLIES
                          if keyword in system), None) or _pick(_GIRL_REPLIES, key)
        else:
            reply = _pick(_GIRL_REPLIES, key)
        return {"event": "chat", "ok": True, "text": json.dumps(reply, ensure_ascii=False)}
    line = _pick(_PLAYER_LINES, key)
    return {"event": "chat", "ok": True, "text": line}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path,
                        default=Path("/tmp/agentic-game-llm-stub.sock"))
    args = parser.parse_args()

    if args.socket.exists():
        args.socket.unlink()
    state = {"calls": 0}
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(args.socket))
    server.listen(8)
    print(f"[dev_llm_stub] listening on {args.socket} (Ctrl-C 停止)")
    try:
        while True:
            conn, _ = server.accept()
            threading.Thread(target=_serve_conn, args=(conn, state),
                             daemon=True).start()
    except KeyboardInterrupt:
        print("\n[dev_llm_stub] bye")
    finally:
        server.close()
        if args.socket.exists():
            args.socket.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
