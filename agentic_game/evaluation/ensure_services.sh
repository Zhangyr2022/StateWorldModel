#!/bin/bash
# 确保评测常驻服务在线;H3 挂掉时按安全顺序重启。
#
# 顺序约定(违反会 OOM,见仓库记忆):H3 装载峰值 ~74GB/卡,LLM 8GiB/卡,
# 必须先停 LLM、再重载 H3、H3 就绪后再起 LLM;严禁并发 start。
#
# 用法:bash evaluation/ensure_services.sh   (幂等,服务在线时秒回)
set -u
cd "$(dirname "$0")/.."

# status 的 ping 只有 5s 超时,worker 忙于渲染时会误报 not running;
# 判死必须连续多次无响应(真死的 worker 永远无响应,忙的会很快回)。
h3_ok() {
  for _ in 1 2 3 4 5; do
    if python3 -m engine.renderer.service status 2>&1 | grep -q "^running"; then
      return 0
    fi
    sleep 15
  done
  return 1
}
llm_ok() { python3 -m engine.llm.service status 2>&1 | grep -q "^running"; }

if ! h3_ok; then
  echo "[ensure] H3 down -> LLM stop, H3 restart, LLM start"
  python3 -m engine.llm.service stop >/dev/null 2>&1
  python3 -m engine.renderer.service stop >/dev/null 2>&1
  python3 -m engine.renderer.service start || { echo "[ensure] H3 start failed"; exit 1; }
  python3 -m engine.llm.service start || { echo "[ensure] LLM start failed"; exit 1; }
elif ! llm_ok; then
  echo "[ensure] LLM down -> LLM start (H3 已驻留,顺序安全)"
  python3 -m engine.llm.service start || { echo "[ensure] LLM start failed"; exit 1; }
fi
h3_ok && llm_ok && echo "[ensure] services ready" || exit 1
