# WBench × ref2va 统一 Prompt 模版(定稿)

> 定稿日期:2026-09-23(六段式修订)。执行器:`wbench_prompt.py`(本文件是语义说明)。
> **本档 = structured(六段式)对照支线的模版定稿,现行默认是 flat 基线句式**
> (逐动作方向句 + 红盒文字,见 `wbench_prompt.py` 的 `FLAT_TEMPLATE`;AB 实测
> 动作无关文本 A/D 会坍缩成前进,方向句必须显式)。六段式依据
> MiniMax-H3 官方 full-reference 写法指南 [`ref-en.txt`](https://github.com/MiniMax-AI/MiniMax-H3/blob/d21241f0a4b3acbb34c97dae47fa417b7065e438/skills/h3-prompt-writing/references/ref-en.txt)
> 第 300-339 行:§7 Complete Example(310-339,六段结构与 retention 固定标记词)
> + 本文末「定稿样例」(跟随/替换句式;原附在 ref-en.txt 副本末尾,指南本身不再随仓库分发)。
> 依据佐证:`h3-a800/prompts/ref2va_*.json`(历代实际送模 prompt,扁平式)与
> `h3-a800/bench/sol-h3-prompts.json`(官方 prompt 写法体系,base 模式 +
> `source_guide` 指向 MiniMax H3 Prompt Writing Guide)。

## 模版全文(六段式,段间空行)

```
<Picture 1> is the exact first frame of the target video: begin the video on <Picture 1>, reproducing it faithfully, and keep its style and lighting for the whole shot. <Subject 1> is {desc}, the character from <Picture 1>. <Video 1> is the motion and camera reference. Replace the blue box with <Subject 1>. <Subject 1> follows the blue box’s motion from <Video 1>, including its body movement, position, direction, timing, and spatial trajectory. The camera follows the camera movement and framing of <Video 1>: {camera}. 

The target video is a single continuous cinematic shot in the same visual style, lighting, color treatment, and scene continuity established by <Picture 1>. Do not preserve the blue box’s placeholder appearance, color, geometry, or material, and let no previz geometry, grid, markers, or placeholder elements remain: the two red reference boxes in <Video 1> are previz-only aids and never appear in the output.

[Shot 1] The shot begins exactly from <Picture 1>. {env} Throughout the clip, the subject keeps its appearance unchanged; the environment layout stays consistent during the move. The clip ends as the camera completes the intended move.

overall_soundscape:
Quiet natural ambience consistent with the scene established by <Picture 1>.

non_diegetic_music:
N/A
```


## 槽位 → WBench 字段映射

六段式**动作无关**:动作只决定渲染哪条控制视频,不进文本(主体运动由
<Video 1> 承载,同 case 各动作 prompt 相同)。槽位仅三个:

| 槽位 | 取值 | 出现次数 | 说明 |
|---|---|---|---|
| `{desc}` | `settings.subject.desc` | ×2(subject_definitions、[Shot 1]) | **纯外观元信息**。官方协议不把它送模型,但"替换蓝盒"任务必须在文本给出主体外观;**不用** `character_prompt`——third_person 的它带姿势/运动尾句("walking forward on the path") |
| `{env}` | `environment_prompt` | ×2(<Subject 2> 定义、[Shot 1] 场景织入) | **整段保留,禁止截断**——`Off-screen ahead/behind/To the left…` 画外句是唯一携带画外空间的文本,直接服务 scene_adherence,并是末段画外揭示段的依据 |
| `{camera}` | `perspective_prompt` | ×1([Shot 1] 构图句) | WBench 视角字段,原样进 |

动作 → 16 动作表校验仍保留(`MOTION`,仅 flat(现行默认)用于逐动作措辞)。

## ref-en.txt 六段式要点落地(§300-339 + 定稿样例)

- **标签**: `<Picture 1>`=首帧锚定(case 原图,`--images-json` → reference 1,
  task type `keyframe completion`);`<Video 1>`=控制视频=运动与运镜参考
  (reference 2,task type `reference generation`)——二者合成 summary 前缀
  `[keyframe completion + reference generation]`(§3 任务类型组合规则)。
  `<Subject 1>`=主体(外观来自 Picture 1、顶替蓝盒)、`<Subject 2>`=真实场景;
  蓝盒**不设** Subject 条目(§2.3:它不是被复用的可见内容,是被替换的占位物,
  其角色在 `<Video 1>` 定义与反残留句中表达)。
- **retention 固定标记词**(§4.1,不可自造):`fully_preserved` /
  `partially_preserved` / `attribute_transfer` / `weak_reference`。本模版四条
  全部 `fully_preserved`:首帧、主体外观、场景、previz 结构的定义角色均被完整
  保留(§4 末注明"新增/变化不算 fidelity 损失"——主体运动按参考变化只写入
  解释句,不降级标记)。
- **detailed_description**:风格句在 `[Shot 1]` 之前(§5.2);单镜头 5s 只写
  `[Shot 1]`(无时间戳,§5.1);锚定用自然短语 "The shot begins from
  <Picture 1>"(§5.3);反残留以陈述句织入("never appear in the output")。
- **音频两段**:WBench 导航 clip 无对白/音乐需求,`overall_soundscape` 给一句
  锚定 <Picture 1> 的自然环境声,`non_diegetic_music` = `N/A`(同 §7 示例)。

## case 字段取舍(对齐上游 WBench prompt_builder 字段)

**进入模版**(3 段自由文本 + 动作键):`environment_prompt`(`<Subject 2>`
定义与 [Shot 1] 场景织入)、`perspective_prompt`([Shot 1] 构图句)、
`settings.subject.desc`(subject_definitions 与 [Shot 1])、
`interactions[].action`(仅决定渲染哪条控制视频;六段式文本不携带动作句,
flat(现行默认)经 `MOTION` 表映射)。

**永不进入**:`settings.scene.*`、`settings.tracking_object`、`settings.style_new`
(风格锚定由 <Picture 1> 承担)、`nav_cate`、`scene_adherence`(env 的人工拆分
版,评测专用)、`subject_adherence`(评测专用)、`metric_list`、
`character_prompt`(被 subject.desc 替代,见上表)。

## 范围与限制

- **仅 third_person 导航 case**(74 个):previz 管线需要可检测主体。
  全部 74 个 case 的动作 ⊆ 16 动作表,`subject.desc` 全部非空(2026-09-23 核验)。
  first_person(84 个)无可见主体,不在本管线范围。
- 6 个混入的非导航轮次(subject_action/event_edit/perspective_switch)不走本模版。
- 每条视频 = 单动作 5s;WBench 多轮逐轮渲染,若做轮间接续,把 `<Picture 1>`
  换成上一段末帧即可,模版语义不变。

## 动作 → 文本关系

- **六段式(structured,对照支线)**:动作不进文本。16 动作仅校验合法性并决定
  控制视频(`clay_subject.py --action`),prompt 文本只由 case 三段槽位决定——
  同 case 各动作 prompt 相同,仍按视频逐 key 输出(h3ref_batch key 约定不变)。
  ⚠️ AB 实测:动作无关文本下 **A/D 横移坍缩成前进**(26/120 case)。
- **flat(现行默认)**:逐动作措辞,`MOTION[act][0]` previz 句 + `MOTION[act][1]`
  主体跟随句,措辞对齐 nav8 定稿(`wbench_prompt.py` 的 `MOTION` 表即定稿措辞)。

## 与 flat(现行默认)的关系

`--style flat` = **现行默认**(nav8 17 条成片验证 + 红盒三处文字;2026-09-23
结构化 4 case 试产 A/D 全坍缩后定为默认),逐字保留于 `FLAT_TEMPLATE`。
两版共用同一 `MOTION` 表与字段取舍,仅组织形式不同;六段式(structured)
留作对照支线,`--self-check` 校验 md 模版槽位齐全可 format。

## 定稿样例(跟随 / 替换句式)

```text
<Subject 1>: The character from <Picture 1>.
Matches the location and the size of the red figure in the first frame of <Video 1> and follows its motion.

<Video 1> is the motion and camera reference.

<Subject 1> follows the motion of the red figure in <Video 1>, including its body movement, position, direction, timing, and spatial trajectory.

The camera follows the camera movement and framing of <Video 1>.

Replace the red figure with <Subject 1>.

Do not preserve the red figure's placeholder appearance or material.

<Picture 1> is the exact first frame of the target video: begin the video on <Picture 1>, reproducing it faithfully, and keep its style and lighting for the whole shot.

<Subject 1> is {desc}, the character from <Picture 1>. 

<Video 1> is the motion and camera reference. Replace the blue box with <Subject 1>. 

<Subject 1> follows the blue box’s motion from <Video 1>, including its body movement, position, direction, timing, and spatial trajectory. The camera follows the camera movement and framing of <Video 1>: {camera}. 

Do not preserve the blue box’s placeholder appearance, color, geometry, or material, and let no previz geometry, grid, markers, or placeholder elements remain. The output is the final polished cinematic shot in the exact style, lighting, and visual identity of <Picture 1>.
```
