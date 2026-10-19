"""VLM 判定:预写问题 → 按帧位提问 → 严格 JSON 回填。

帧位契约:Picture 1 = 首帧(= input.png 的渲染复现),其后按时间序采样,
末图 = 末帧。initial 问题只看 Picture 1(前提是否成立),action 问题看
全程,outcome 问题只看结尾(分支结局)。答案必须逐字取自数据给出的
answers 选项;解析不出/越界的答案记 None(指标按错答计,不猜)。

- LocalJudge:本进程 transformers 直载判定 VLM(默认 config.review.model,
  与数据生产侧 review.py 同一加载路径;单卡,贪心解码,思考模式关);
- HeuristicJudge:直接回填 expected,仅用于无 GPU 打通数据流,绝不能当
  正式结果。

并行方式与数据生产侧一致:按卡分片子进程(--gpus 0,1,2 … 每卡一个
worker 进程裁一份数据),见 worker.py / run_eval.py,不再是原
agentic_game 版的常驻 socket 机群。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from evaluation.dataset import TwinSpec

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


class LocalJudge:
    """本进程 transformers 判定(模型加载沿用 review.py 的同一路径)。

    多图输入:消息 content = N 个 image 项 + 文本题面;贪心解码,
    思考模式关(与 review.py 相同的 enable_thinking 兼容回退)。
    """

    name = "local"

    def __init__(self, *, model: str | None = None, device: str = "cuda:0",
                 seed: int = 0, max_tokens: int = 768) -> None:
        import torch  # noqa: PLC0415 - 重依赖只在实例化时导入
        from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

        from worldstate_bench.config import load_config  # noqa: PLC0415

        self._torch = torch
        model_path = model or load_config().review.get("model")
        if not model_path:
            raise SystemExit("config.yaml 缺少 review.model,且未传 --model")
        self._processor = AutoProcessor.from_pretrained(model_path)
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device)
        self._model.eval()
        self._device = device
        self._seed = seed
        self._max_tokens = max_tokens

    def judge(self, twin: TwinSpec, images: list[Path]) -> dict:
        torch = self._torch
        content = [{"type": "image", "image": str(path)} for path in images]
        content.append({"type": "text",
                        "text": build_user_prompt(twin, len(images))})
        messages = [
            {"role": "system", "content": TASKA_JUDGE_SYS},
            {"role": "user", "content": content},
        ]
        try:
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:  # 模板不支持该参数的判定器
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        pil_images = self._load_images(images)
        inputs = self._processor(text=[text], images=pil_images,
                                 return_tensors="pt").to(self._device)
        with torch.no_grad():
            output = self._model.generate(**inputs,
                                          max_new_tokens=self._max_tokens,
                                          do_sample=False)
        raw = self._processor.decode(output[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
        return parse_verdict(raw, twin)

    def _load_images(self, images: list[Path]) -> list:
        from PIL import Image  # noqa: PLC0415

        return [Image.open(path).convert("RGB") for path in images]

    def close(self) -> None:
        return None


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


def build_judge(name: str, *, model: str | None = None, device: str = "cuda:0",
                seed: int = 0, max_tokens: int = 768):
    if name == "heuristic":
        return HeuristicJudge()
    if name == "local":
        return LocalJudge(model=model, device=device, seed=seed,
                          max_tokens=max_tokens)
    raise ValueError(f"unknown judge backend: {name!r} (heuristic | local)")


__all__ = ["TASKA_JUDGE_SYS", "HeuristicJudge", "LocalJudge",
           "build_judge", "build_user_prompt", "parse_verdict"]
