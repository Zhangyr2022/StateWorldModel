"""h3_worker 请求协议归一测试(双协议:engine + box2clay ref2va_serve)。

只测纯函数 normalize_request,不加载引擎(§25)。
"""

import unittest

from engine.renderer.h3_worker import SERVICE_TAG, normalize_request


class NormalizeRequestTest(unittest.TestCase):
    def test_engine_protocol_passthrough(self):
        request = {"prompt": "p", "seed": 7, "duration_s": 5,
                   "first_frame": "/tmp/a.png", "reference_video": "/tmp/b.mp4",
                   "output": "/tmp/out.mp4", "turn_id": "turn-001"}
        self.assertEqual(normalize_request(request), request)

    def test_box2clay_protocol_mapped(self):
        raw = {"cmd": "generate", "prompt": "p", "video": "/tmp/b.mp4",
               "image": "/tmp/a.png", "duration": 10, "seed": 42,
               "out": "/tmp/out.mp4"}
        normalized = normalize_request(raw)
        self.assertEqual(normalized["prompt"], "p")
        self.assertEqual(normalized["seed"], 42)
        self.assertEqual(normalized["duration_s"], 10)
        self.assertEqual(normalized["first_frame"], "/tmp/a.png")
        self.assertEqual(normalized["reference_video"], "/tmp/b.mp4")
        self.assertEqual(normalized["output"], "/tmp/out.mp4")
        self.assertNotIn("video", normalized)      # 不与引擎键混淆
        self.assertNotIn("cmd", normalized)

    def test_box2clay_without_image(self):
        raw = {"cmd": "generate", "prompt": "p", "video": "/tmp/b.mp4",
               "out": "/tmp/out.mp4"}
        normalized = normalize_request(raw)
        self.assertIsNone(normalized["first_frame"])
        self.assertEqual(normalized["duration_s"], 5)  # box2clay 默认
        self.assertEqual(normalized["seed"], 0)

    def test_service_tag_exists_for_pong_marker(self):
        # runner 依赖 pong 里的 service 标记区分 engine worker 与
        # box2clay 原生 ref2va_serve(后者 pong 无此键)
        self.assertTrue(SERVICE_TAG)


if __name__ == "__main__":
    unittest.main()
