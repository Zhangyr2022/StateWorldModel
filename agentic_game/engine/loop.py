"""GameLoop:Turn System——把 Runtime、Visual Pipeline、Renderer 串成回合。

顺序即架构(论文 §5:Decision → Transition → Realization):
    玩家选择 → 确定性结算 → 更新 Game State → Visual Script → Prompt
    → 视频 → 最后一帧 → 下一回合

关键约束:
- 贡献 1(Rule-Grounded):结算永远先于渲染;Renderer 失败属于表现层
  失败,不允许影响 Game State(见 AGENTS.md「Engine engineering rules」):跳过视觉或
  中止,绝不回滚/不确定化状态。
- 贡献 3(Agent-Mediated):agent.choose 是玩家意图进入系统的唯一入口,
  选择逻辑可以是 CLI、脚本、随机或 LLM,规则层完全无感知;post_render
  钩子允许游戏层的 VLM 观看视频后参与下一回合提议。
- 贡献 2/4:本模块不知道任何具体游戏规则与 prompt 语法——游戏通过
  GameRuntime 协议 + visual_director 回调接入,渲染经 Renderer 抽象。
- 上一段视频的最后一帧作为下一段的首帧(末帧链接约定)。
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from engine.agents import PlayerAgent
from engine.renderer.base import (
    InvalidRenderRequestError,
    RenderError,
    RenderFailedError,
    RenderRequest,
    RenderResult,
    Renderer,
)
from engine.runtime import (
    STATUS_ONGOING,
    Action,
    GameRuntime,
    Rng,
    TurnOutcome,
    TurnResult,
)
from engine.visual.compiler import PromptCompiler
from engine.visual.script import VisualScript
from engine.visual.state import VisualState

logger = logging.getLogger("engine.loop")

# 游戏层提供的 TurnResult → VisualScript 映射函数。
VisualDirector = Callable[[TurnResult, VisualState], VisualScript]
# 游戏层提供的结束动画映射:输入终局状态(won/lost/draw),返回演出描述;
# 返回 None 表示该结局不生成动画。
EndingDirector = Callable[[str, VisualState], VisualScript | None]
# 游戏层提供的控制视频生产方:script.reference 非 None 时被调用,返回
# 控制视频文件路径(生产/命中缓存均由游戏层负责;典型实现 = 检测 + 白模
# previz 渲染)。返回的路径进入 RenderRequest.reference_video,并决定
# prompt 是否带 <Video 1>。生产失败抛 RenderError 子类,由回合按
# render_failure_policy 处理(表现层失败,不影响 Game State)。
ReferenceProducer = Callable[[VisualScript, VisualState], Path]

RENDER_FAILURE_ABORT = "abort"  # 渲染失败时中止整个游戏
RENDER_FAILURE_SKIP = "skip"    # 渲染失败时记录并沿用上一帧继续(状态不变)


@dataclass
class TurnRecord:
    """一个回合的落地结果,用于日志与 UI 摘要。"""

    turn_id: str
    index: int
    action_id: str
    status: str
    prompt: str = ""
    caption: str = ""  # 该回合结束时的一行状态摘要(供回放字幕)
    meta: dict = field(default_factory=dict)  # 游戏私有元数据(如场景 id),engine 不解释
    video_path: str | None = None
    final_frame: str | None = None
    reference_video: str | None = None  # 本回合携带的控制视频(白模 previz)
    render_error: str | None = None
    render_elapsed_s: float = 0.0
    seed: int = 0


# 渲染后钩子:每段视频成功产出后调用(传入本回合记录),游戏层可用它
# 实现如"VLM 观看视频并准备下一回合的选项"这类逻辑。钩子异常被吞掉并记日志。
PostRenderHook = Callable[["TurnRecord"], None]


@dataclass
class RunSummary:
    """一局游戏的汇总。"""

    session_dir: Path
    records: list[TurnRecord] = field(default_factory=list)
    ending: TurnRecord | None = None  # 结束动画回合(如提供了 ending_director)
    final_status: str = STATUS_ONGOING
    turns_played: int = 0
    finished_reason: str = ""  # game_over / max_turns / no_actions

    @property
    def ok(self) -> bool:
        return all(record.render_error is None for record in self.records)


class GameLoop:
    def __init__(
        self,
        runtime: GameRuntime,
        agent: PlayerAgent,
        visual_director: VisualDirector,
        renderer: Renderer,
        *,
        output_dir: Path,
        base_seed: int = 0,
        max_turns: int = 32,
        render_retries: int = 1,
        render_failure_policy: str = RENDER_FAILURE_SKIP,
        visual_state: VisualState | None = None,
        compiler: PromptCompiler | None = None,
        ending_director: EndingDirector | None = None,
        post_render: PostRenderHook | None = None,
        hud_of: Callable[[object], dict] | None = None,
        reference_producer: ReferenceProducer | None = None,
    ) -> None:
        if render_failure_policy not in (RENDER_FAILURE_ABORT, RENDER_FAILURE_SKIP):
            raise ValueError(f"unknown render_failure_policy: {render_failure_policy}")
        self.runtime = runtime
        self.agent = agent
        self.visual_director = visual_director
        self.renderer = renderer
        self.reference_producer = reference_producer
        self.compiler = compiler or PromptCompiler()
        self.ending_director = ending_director
        self.post_render = post_render
        self.hud_of = hud_of
        self.visual_state = visual_state or VisualState()
        self.base_seed = base_seed
        self.max_turns = max_turns
        self.render_retries = render_retries
        self.render_failure_policy = render_failure_policy
        self.output_dir = Path(output_dir)
        self.rng = Rng(base_seed)  # 唯一随机源,整局共享保证确定性
        self._log_path = self.output_dir / "turn_log.jsonl"

    # -- 主循环 --------------------------------------------------------------

    def run(self) -> RunSummary:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        state = self.runtime.initial_state()
        summary = RunSummary(session_dir=self.output_dir)

        while True:
            status = self.runtime.status_of(state)
            if status != STATUS_ONGOING:
                summary.finished_reason = "game_over"
                summary.final_status = status
                break
            if len(summary.records) >= self.max_turns:
                summary.finished_reason = "max_turns"
                break
            actions = self.runtime.available_actions(state)
            if not actions:
                summary.finished_reason = "no_actions"
                break

            action = self.agent.choose(state, actions)
            record, state = self.run_turn(state, action, index=len(summary.records))
            summary.records.append(record)

        summary.turns_played = len(summary.records)
        if summary.finished_reason == "game_over" and self.ending_director is not None:
            # 结束动画:一段独立于回合的终局演出,同样只表现已确定的结果。
            summary.ending = self.run_ending(summary.final_status)
        logger.info(
            "game finished: reason=%s status=%s turns=%d dir=%s",
            summary.finished_reason,
            summary.final_status,
            summary.turns_played,
            self.output_dir,
        )
        return summary

    def run_turn(self, state: Any, action: Action, index: int) -> tuple[TurnRecord, Any]:
        """执行一个回合:结算 → Visual Script → Prompt → 渲染 → 记录。

        返回 (回合记录, 结算后的新状态)。
        """
        turn_id = f"turn-{index + 1:03d}"
        state_before = state

        # 1. 确定性结算:游戏结果在此刻已完全确定,与渲染成败无关。
        outcome = self.runtime.resolve_turn(state_before, action, self.rng)
        result = TurnResult(
            turn_id=turn_id,
            index=index,
            action=action,
            state_before=state_before,
            state_after=outcome.state_after,
            outcome=outcome,
        )
        logger.info("[%s] action=%s status=%s events=%s", turn_id, action.id, outcome.status,
                    [event.type for event in outcome.events])

        # 2. 游戏层把结果翻译成视觉描述;如声明了控制视频(MotionReference)
        #    先由 reference_producer 生产/命中缓存——它决定 prompt 是否带
        #    <Video 1>(消融与降级的切换点)。生产失败属于表现层失败,
        #    按 render_failure_policy 处理,不影响上面的结算。
        script = self.visual_director(result, self.visual_state)
        reference_video: Path | None = None
        if script.reference is not None:
            reference_video = self._produce_reference(script, result, turn_id)
        # 带首帧(I2VA)与否决定 prompt 的对齐指令;script.use_first_frame
        # 为 False 时本回合走 t2v(如场景切换回合)。
        has_first_frame = (
            self.visual_state.current_frame is not None and script.use_first_frame
        )
        prompt = self.compiler.compile(
            script,
            has_first_frame=has_first_frame,
            has_reference_video=reference_video is not None,
        )

        # 3. 渲染:结果已定,这里只是"演出"。失败不影响上面的结算。
        record = TurnRecord(
            turn_id=turn_id,
            index=index,
            action_id=action.id,
            status=outcome.status,
            prompt=prompt,
            caption=f"{action.label} | {self.runtime.describe_state(outcome.state_after)}",
            meta=dict(script.meta),
            seed=self.base_seed + index,
            reference_video=str(reference_video) if reference_video else None,
        )
        if self.hud_of is not None:
            # HUD 快照只服务回放烧录,属于表现层元数据。
            hud_snapshot = self.hud_of(outcome.state_after, outcome.events)
            hud_snapshot.setdefault("action", action.label)
            hud_snapshot.setdefault("action_desc", action.description)
            hud_snapshot.setdefault("tag", f"第{index + 1}回合 · {action.label}")
            record.meta["hud"] = hud_snapshot
        render_result, render_error = self._render_with_retries(
            RenderRequest(
                turn_id=turn_id,
                prompt=prompt,
                seed=record.seed,
                first_frame=self.visual_state.current_frame if has_first_frame else None,
                duration_s=script.duration_s,
                reference_video=reference_video,
            )
        )
        if render_result is not None:
            record.video_path = str(render_result.video_path)
            record.final_frame = str(render_result.final_frame)
            record.render_elapsed_s = render_result.elapsed_s
            # 画面连续性:本段视频的最后一帧是下一段的首帧(所有模式统一,
            # 含 ref2va 控制视频——生产方按当前帧重新检测/生成白模)。
            self.visual_state.current_frame = render_result.final_frame
            if self.post_render is not None:
                # 渲染后的游戏层钩子:如开放世界游戏用 VLM 观看本段视频、
                # 为下一回合提议行动。钩子失败只记录,不影响回合记录。
                try:
                    self.post_render(record)
                except Exception as exc:  # noqa: BLE001 - 钩子是增益,不是规则
                    logger.warning("[%s] post_render hook failed: %s", turn_id, exc)
        else:
            record.render_error = str(render_error)
            if self.render_failure_policy == RENDER_FAILURE_ABORT:
                self._write_log_line(result, script, record)
                raise render_error
            logger.warning(
                "[%s] render failed (%s); continuing with previous frame; "
                "game state is unaffected",
                turn_id,
                render_error,
            )

        # 4. 记录与落盘(§17/§18:一个 turn 关联全部产物,可追溯)。
        self.visual_state.turn_index = index + 1
        self._write_log_line(result, script, record)
        return record, result.state_after

    # -- 结束动画 ------------------------------------------------------------

    def run_ending(self, status: str) -> TurnRecord | None:
        """渲染终局演出(如游戏层提供;结算早已完成,这里同样是纯表现)。"""
        assert self.ending_director is not None
        script = self.ending_director(status, self.visual_state)
        if script is None:
            return None

        turn_id = f"ending-{status}"
        prompt = self.compiler.compile(
            script, has_first_frame=self.visual_state.current_frame is not None
        )
        record = TurnRecord(
            turn_id=turn_id,
            index=self.max_turns,
            action_id="ending",
            status=status,
            prompt=prompt,
            seed=self.base_seed + 997,
        )
        render_result, render_error = self._render_with_retries(
            RenderRequest(
                turn_id=turn_id,
                prompt=prompt,
                seed=record.seed,
                first_frame=self.visual_state.current_frame,
                duration_s=script.duration_s,
            )
        )
        if render_result is not None:
            record.video_path = str(render_result.video_path)
            record.final_frame = str(render_result.final_frame)
            record.render_elapsed_s = render_result.elapsed_s
        else:
            record.render_error = str(render_error)
            if self.render_failure_policy == RENDER_FAILURE_ABORT:
                self._write_log_line(_ending_turn_result(status), script, record)
                raise render_error
            logger.warning("[%s] ending render failed (%s); game result is unaffected",
                           turn_id, render_error)
        self._write_log_line(_ending_turn_result(status), script, record)
        return record

    # -- 渲染与日志 ----------------------------------------------------------

    def _produce_reference(
        self, script: VisualScript, result: TurnResult, turn_id: str
    ) -> Path | None:
        """生产/命中控制视频;失败按 render_failure_policy 处理(表现层)。

        返回 None 表示本回合不带控制视频继续(prompt 自动切换到无视频
        变体);abort 策略下失败即中止整局,与渲染失败同语义。
        """
        if self.reference_producer is None:
            logger.warning(
                "[%s] script declares MotionReference but no reference_producer "
                "was provided; compiling without the control video",
                turn_id,
            )
            return None
        try:
            video = Path(self.reference_producer(script, self.visual_state))
        except RenderError as exc:
            return self._reference_failed(script, result, turn_id, exc)
        except Exception as exc:  # noqa: BLE001 - 生产方异常统一归类为表现层失败
            return self._reference_failed(
                script, result, turn_id,
                RenderFailedError(f"reference video production failed: {exc}"),
            )
        if not video.is_file():
            return self._reference_failed(
                script, result, turn_id,
                RenderFailedError(
                    f"reference producer returned a missing file: {video}"
                ),
            )
        return video

    def _reference_failed(
        self, script: VisualScript, result: TurnResult, turn_id: str, exc: RenderError
    ) -> Path | None:
        if self.render_failure_policy == RENDER_FAILURE_ABORT:
            record = TurnRecord(
                turn_id=turn_id,
                index=result.index,
                action_id=result.action.id,
                status=result.status,
                render_error=str(exc),
            )
            self._write_log_line(result, script, record)
            raise exc
        logger.warning(
            "[%s] reference video production failed (%s); continuing without "
            "the control video; game state is unaffected",
            turn_id, exc,
        )
        return None


    def _render_with_retries(
        self, request: RenderRequest
    ) -> tuple[RenderResult | None, RenderError | None]:
        attempts = max(1, self.render_retries + 1)
        last_error: RenderError | None = None
        for attempt in range(1, attempts + 1):
            try:
                return self.renderer.render(request), None
            except InvalidRenderRequestError:
                raise  # 请求非法,重试没有意义
            except RenderError as exc:
                last_error = exc
                logger.warning(
                    "[%s] render attempt %d/%d failed: %s",
                    request.turn_id,
                    attempt,
                    attempts,
                    exc,
                )
        return None, last_error

    def _write_log_line(self, result: TurnResult, script: VisualScript, record: TurnRecord) -> None:
        entry = {
            "turn_id": result.turn_id,
            "index": result.index,
            "action": {"id": result.action.id, "label": result.action.label},
            "status": result.status,
            "state_before": _to_jsonable(result.state_before),
            "state_after": _to_jsonable(result.state_after),
            "events": [event.to_dict() for event in result.events],
            "visual_script": script.to_dict(),
            "prompt": record.prompt,
            "render": {
                "seed": record.seed,
                "video": record.video_path,
                "final_frame": record.final_frame,
                "reference_video": record.reference_video,
                "elapsed_s": record.render_elapsed_s,
                "error": record.render_error,
            },
        }
        with open(self._log_path, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _ending_turn_result(status: str) -> TurnResult:
    """给结束动画合成一个最小 TurnResult,复用回合日志通道。"""
    return TurnResult(
        turn_id=f"ending-{status}",
        index=-1,
        action=Action(id="ending", label="结束动画"),
        state_before=None,
        state_after=None,
        outcome=TurnOutcome(events=(), status=status),
    )


def _to_jsonable(obj: Any) -> Any:
    """dataclass → dict;实现 to_snapshot() 的状态对象 → 结构化快照;
    其余对象走 repr 兜底,保证日志总能落盘。"""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            key: _to_jsonable(value)
            for key, value in dataclasses.asdict(obj).items()
        }
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    snapshot = getattr(obj, "to_snapshot", None)
    if callable(snapshot):
        return snapshot()
    if isinstance(obj, dict):
        return {str(key): _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(item) for item in obj]
    return repr(obj)
