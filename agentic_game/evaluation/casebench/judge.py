"""VLM 判定:预写问题 → 按帧位提问 → 严格 JSON 回填。

帧位契约:Picture 1 = 首帧(= input.png 的渲染复现),其后按时间序采样,
末图 = 末帧。initial 问题只看 Picture 1(前提是否成立),action 问题看
全程,outcome 问题只看结尾(分支结局)。答案必须逐字取自数据给出的
answers 选项;解析不出/越界的答案记 None(指标按错答计,不猜)。

- ResidentJudge:engine 常驻 Qwen VL 服务(Unix socket,贪心解码);
- HeuristicJudge:直接回填 expected,仅用于无 GPU 打通数据流,绝不能当
  正式结果。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from evaluation.casebench.dataset import TwinSpec

TASKA_JUDGE_SYS = (
    "You are a precise video reviewer. You answer questions about a single "
    "continuous clip using ONLY what is visible in the provided frames. "
    "Frames arrive as numbered pictures in chronological order: Picture 1 is "
    "the first frame of the clip and the last picture is the final frame. "
    "Report what the clip actually shows, not what it was supposed to show; "
    "if the visual evidence is ambiguous or absent, pick the 'unclear' option "
    "when one exists. Copy every answer verbatim from the given options. "
    "Reply with exactly one JSON object and nothing else."
)


def build_user_prompt(twin: TwinSpec, n_images: int) -> str:
    """把数据预写的问题排版成判定提示(问题原文一字不改)。

    counter 问题由 terminal_visual_states.not_occurs 机械派生(数据给短语,
    代码只排版):严格结局口径要求反分支结果在全片任何时刻都不出现。
    """
    lines = [
        f"You are reviewing {n_images} pictures sampled in chronological order "
        "from one continuous clip (Picture 1 = first frame, last picture = "
        "final frame).",
        "",
        "Initial-state questions — answer from Picture 1 ONLY:",
    ]
    for index, question in enumerate(twin.initial_questions, start=1):
        lines.append(f"  I{index}. {question['question']} "
                     f"Options: {_options(question['answers'])}")
    lines += [
        "",
        f"Action question — answer from the whole clip: "
        f"{twin.action_question['question']} "
        f"Options: {_options(twin.action_question['answers'])}",
        "",
        f"Outcome question — answer from the END of the clip: "
        f"{twin.outcome_question['question']} "
        f"Options: {_options(twin.outcome_question['answers'])}",
        "",
        f"Counter-outcome question — answer from the WHOLE clip: "
        f"{twin.counter_question['question']} "
        f"Options: {_options(twin.counter_question['answers'])}",
        "",
        "Reply with exactly one JSON object in this exact shape:",
        '{"initial": ["<answer to I1>", "<answer to I2>", ...], '
        '"action": "<answer>", "outcome": "<answer>", "counter": "<answer>", '
        '"evidence": "<one short sentence>"}',
        f"- 'initial' must contain exactly {len(twin.initial_questions)} answers, "
        "in the I1..In order.",
        "- every answer must be copied verbatim from that question's options.",
    ]
    return "\n".join(lines)


def parse_verdict(raw: str, twin: TwinSpec) -> dict:
    """模型输出 → 规范 verdict;解析失败抛 ValueError(上层记失败,不猜)。"""
    matched = re.search(r"\{.*\}", raw, re.DOTALL)
    if not matched:
        raise ValueError(f"judge output has no JSON object: {raw[:200]!r}")
    data = json.loads(matched.group(0))
    if not isinstance(data, dict):
        raise ValueError(f"judge output is not a JSON object: {raw[:200]!r}")
    initial_raw = data.get("initial")
    if not isinstance(initial_raw, list):
        raise ValueError(f"judge verdict missing 'initial' list: {raw[:200]!r}")
    initial = [_canon(answer, twin.initial_questions[index]["answers"])
               if index < len(twin.initial_questions) else None
               for index, answer in enumerate(initial_raw)]
    return {
        "initial": initial,
        "action": _canon(data.get("action"), twin.action_question["answers"]),
        "outcome": _canon(data.get("outcome"), twin.outcome_question["answers"]),
        "counter": _canon(data.get("counter"), twin.counter_question["answers"]),
        "evidence": str(data.get("evidence", "")).strip(),
    }


def _options(answers: list) -> str:
    return " / ".join(f"'{answer}'" for answer in answers)


def _canon(answer: object, allowed: list) -> str | None:
    """答案对齐到选项原文(大小写/空白不敏感);越界即 None,不猜。"""
    if not isinstance(answer, str):
        return None
    stripped = answer.strip()
    for option in allowed:
        if stripped.lower() == str(option).strip().lower():
            return str(option)
    return None


# -- 后端 --------------------------------------------------------------------


class ResidentJudge:
    """engine 常驻 Qwen VL 服务(与渲染共存,GPU 上贪心解码)。"""

    name = "resident"

    def __init__(self, *, socket_path: Path | None = None, seed: int = 0,
                 max_tokens: int = 768, timeout_s: float = 600.0,
                 ready_timeout_s: float = 60.0) -> None:
        from engine.llm.client import LLMClient
        from engine.llm.service import default_llm_socket

        self._client = LLMClient(
            socket_path=socket_path or Path(
                os.environ.get("AGENTIC_GAME_LLM_SOCKET") or default_llm_socket()),
            generate_timeout_s=timeout_s,
        )
        if self._client.ping() is None:
            self._client.wait_until_ready(timeout_s=ready_timeout_s)
        self._seed = seed
        self._max_tokens = max_tokens

    def judge(self, twin: TwinSpec, images: list[Path]) -> dict:
        text = self._client.chat(
            messages=[
                {"role": "system", "content": TASKA_JUDGE_SYS},
                {"role": "user", "content": build_user_prompt(twin, len(images))},
            ],
            seed=self._seed,
            max_tokens=self._max_tokens,
            images=list(images),
        )
        return parse_verdict(text, twin)

    def close(self) -> None:
        self._client.call_count = 0  # 无持久连接可关;保留接口对称


class ParallelJudge:
    """N 路单卡 LLM worker 并行判定(evaluation.casebench.fleet 起停)。

    judge() 从空闲队列借一个单 worker 客户端用完即还;外层用
    ThreadPoolExecutor 把 clip 摊到 N 路,耗时约除以 N。每个 LLMClient
    每次调用新建连接,线程间无共享状态。
    """

    name = "parallel"

    def __init__(self, sockets: list[Path], *, seed: int = 0,
                 max_tokens: int = 768, timeout_s: float = 600.0) -> None:
        import queue as _queue

        if len(sockets) < 2:
            raise ValueError("ParallelJudge needs at least two sockets")
        self._judges: list[ResidentJudge] = [
            ResidentJudge(socket_path=socket, seed=seed, max_tokens=max_tokens,
                          timeout_s=timeout_s, ready_timeout_s=5.0)
            for socket in sockets
        ]
        self._idle: _queue.Queue[ResidentJudge] = _queue.Queue()
        for judge in self._judges:
            self._idle.put(judge)

    def judge(self, twin: TwinSpec, images: list[Path]) -> dict:
        judge = self._idle.get()
        try:
            return judge.judge(twin, images)
        finally:
            self._idle.put(judge)

    def close(self) -> None:
        for judge in self._judges:
            judge.close()


class HeuristicJudge:
    """干跑后端:直接回填 expected(有采样帧才给判定)。

    用途仅限"打通流水线/离线测试"——它没有看视频,产出绝不能当正式结果。
    """

    name = "heuristic"

    def judge(self, twin: TwinSpec, images: list[Path]) -> dict:
        if not images:
            raise ValueError("heuristic judge needs sampled frames")
        return {
            "initial": [str(question["expected"]) for question in twin.initial_questions],
            "action": str(twin.action_question["expected"]),
            "outcome": str(twin.outcome_question["expected"]),
            "counter": str(twin.counter_question["expected"]),
            "evidence": "heuristic dry-run: follows data expectations",
        }

    def close(self) -> None:
        return None


def build_judge(name: str, *, socket_path: Path | None = None, seed: int = 0,
                max_tokens: int = 768, sockets: list[Path] | None = None):
    if name == "heuristic":
        return HeuristicJudge()
    if name == "resident":
        return ResidentJudge(socket_path=socket_path, seed=seed, max_tokens=max_tokens)
    if name == "parallel":
        if not sockets:
            raise ValueError("parallel judge needs fleet sockets")
        return ParallelJudge(sockets, seed=seed, max_tokens=max_tokens)
    raise ValueError(f"unknown judge backend: {name!r} (heuristic | resident | parallel)")


__all__ = ["TASKA_JUDGE_SYS", "HeuristicJudge", "ParallelJudge", "ResidentJudge",
           "build_judge", "build_user_prompt", "parse_verdict"]
