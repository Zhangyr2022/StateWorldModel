"""GPU 卡数自动探测测试:nproc <= 0 时常驻服务有多少可见卡就用多少卡。

覆盖 engine/config.py 的 detect_gpu_count(CUDA_VISIBLE_DEVICES 语义、
nvidia-smi 兜底)与 load_config 的单点解析(配置/环境变量的优先级)。
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from engine.config import detect_gpu_count, load_config


def _write_config(tmp: str, nproc_line: str) -> Path:
    path = Path(tmp) / "config.toml"
    path.write_text(f"[renderer]\n{nproc_line}\n", encoding="utf-8")
    return path


class DetectGpuCountTest(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    def tearDown(self):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        if self._saved is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self._saved

    def test_counts_cuda_visible_devices(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"
        self.assertEqual(detect_gpu_count(), 3)  # 不触发 nvidia-smi

    def test_ignores_blank_entries(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, ,2"
        self.assertEqual(detect_gpu_count(), 2)

    def test_empty_visible_means_zero_gpus(self):
        # CUDA 语义:设了空串 = 隐藏全部 GPU(不回落 nvidia-smi)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self.assertEqual(detect_gpu_count(), 0)

    def test_missing_nvidia_smi_returns_zero(self):
        with mock.patch("engine.config.subprocess.run", side_effect=OSError("no cmd")):
            self.assertEqual(detect_gpu_count(), 0)


class LoadConfigNprocTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = {k: os.environ.get(k)
                       for k in ("CUDA_VISIBLE_DEVICES", "AGENTIC_GAME_NPROC")}
        os.environ.pop("AGENTIC_GAME_NPROC", None)

    def tearDown(self):
        self._tmp.cleanup()
        for key, value in self._saved.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value

    def test_zero_means_autodetect(self):
        path = _write_config(self._tmp.name, "nproc = 0")
        with mock.patch("engine.config.detect_gpu_count", return_value=5):
            self.assertEqual(load_config(path).renderer.nproc, 5)

    def test_positive_pins_card_count(self):
        path = _write_config(self._tmp.name, "nproc = 3")
        with mock.patch("engine.config.detect_gpu_count", return_value=9) as det:
            self.assertEqual(load_config(path).renderer.nproc, 3)
            det.assert_not_called()  # 显式配置不探测

    def test_env_var_wins_over_file(self):
        path = _write_config(self._tmp.name, "nproc = 0")
        os.environ["AGENTIC_GAME_NPROC"] = "2"
        with mock.patch("engine.config.detect_gpu_count", return_value=9):
            self.assertEqual(load_config(path).renderer.nproc, 2)

    def test_no_gpu_falls_back_to_one(self):
        with mock.patch("engine.config.detect_gpu_count", return_value=0):
            self.assertEqual(load_config(Path(self._tmp.name) / "absent.toml")
                             .renderer.nproc, 1)


if __name__ == "__main__":
    unittest.main()
