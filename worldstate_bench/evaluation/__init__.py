"""WorldState-Bench evaluation package.

Consumes the packed dataset (``out/cases/<CASE>/``, produced by ``run.py
package``) and judges generated clips with a VLM:

    dataset.py   load typed twin cases from the dataset
    frames.py    video -> sampled frames for the judge (ffmpeg)
    judge.py     pre-written questions -> frame-positioned VLM prompt -> JSON verdict
    metrics.py   verdict + gold -> clip metrics (ENDFOLLOW / CNTABSENT / STRICT)
    report.py    verdicts -> metrics.json + report.md + FAILURES.md
    headless.py  zero-GPU dataset self-consistency checks (rule replay via the repo DSL)
    rules.py     deterministic rule replay (reuses worldstate_bench.dsl)
    worker.py    single-GPU judge shard CLI
    run_eval.py  orchestrator: headless -> materialize -> judge -> report

Entry points:
    python3 -m evaluation.run_eval --help
"""
