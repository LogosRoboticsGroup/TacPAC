from __future__ import annotations

import unittest
from io import BytesIO

import numpy as np
from PIL import Image

from deployment.model_server.infersystem_protocol import (
    InferSystemAdapterConfig,
    InferSystemProtocolError,
    extract_action_chunk,
    infersystem_to_starvla_payload,
    msgpack_safe,
    parse_camera_order,
)


def _jpeg_bytes(color: tuple[int, int, int], size: tuple[int, int] = (8, 8)) -> bytes:
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    image[:, :] = np.asarray(color, dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(image, mode="RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue()


class InferSystemProtocolTest(unittest.TestCase):
    def test_converts_predict_request_in_configured_camera_order(self) -> None:
        config = InferSystemAdapterConfig(
            camera_order=parse_camera_order("wrist,main"),
            stat_key="arx_x5",
            default_fps=30,
        )
        payload = infersystem_to_starvla_payload(
            {
                "cmd": "predict",
                "prompt": "pick up the cup",
                "state": [1.0, 2.0, 3.0],
                "main": _jpeg_bytes((255, 0, 0)),
                "wrist": _jpeg_bytes((0, 255, 0)),
            },
            config,
        )

        self.assertEqual(payload["instructions"], ["pick up the cup"])
        self.assertEqual(payload["stat_key"], "arx_x5")
        self.assertEqual(payload["fps"], [30.0])
        self.assertEqual(payload["view_mask"], [[True, True]])
        self.assertNotIn("camera_order", payload)
        self.assertEqual(np.asarray(payload["state"][0]).shape, (3,))
        self.assertGreater(payload["batch_images"][0][0][0, 0, 1], payload["batch_images"][0][0][0, 0, 0])
        self.assertGreater(payload["batch_images"][0][1][0, 0, 0], payload["batch_images"][0][1][0, 0, 1])

    def test_pre_resizes_camera_images_before_policy_preprocess(self) -> None:
        config = InferSystemAdapterConfig(camera_order=("main",), image_resize_size=(224, 224))
        payload = infersystem_to_starvla_payload(
            {
                "cmd": "predict",
                "state": [0.0],
                "main": _jpeg_bytes((255, 0, 0), size=(640, 480)),
            },
            config,
        )

        self.assertEqual(payload["batch_images"][0][0].shape, (224, 224, 3))

    def test_strict_cameras_rejects_missing_and_unexpected_views(self) -> None:
        config = InferSystemAdapterConfig(camera_order=("main", "wrist"))
        with self.assertRaisesRegex(InferSystemProtocolError, "missing camera"):
            infersystem_to_starvla_payload({"cmd": "predict", "main": _jpeg_bytes((1, 2, 3))}, config)

        with self.assertRaisesRegex(InferSystemProtocolError, "unexpected camera"):
            infersystem_to_starvla_payload(
                {
                    "cmd": "predict",
                    "main": _jpeg_bytes((1, 2, 3)),
                    "wrist": _jpeg_bytes((4, 5, 6)),
                    "side": _jpeg_bytes((7, 8, 9)),
                },
                config,
            )

    def test_missing_camera_is_not_filled(self) -> None:
        config = InferSystemAdapterConfig(
            camera_order=("main", "wrist"),
            strict_cameras=False,
        )
        with self.assertRaisesRegex(InferSystemProtocolError, "missing camera"):
            infersystem_to_starvla_payload({"cmd": "predict", "main": _jpeg_bytes((1, 2, 3))}, config)

    def test_extra_can_override_stat_key_and_fps(self) -> None:
        config = InferSystemAdapterConfig(camera_order=("main",), stat_key="default_key")
        payload = infersystem_to_starvla_payload(
            {
                "cmd": "predict",
                "main": _jpeg_bytes((1, 2, 3)),
                "extra": {"stat_key": "runtime_key", "fps": 24, "use_ddim": True},
            },
            config,
        )

        self.assertEqual(payload["stat_key"], "runtime_key")
        self.assertEqual(payload["fps"], [24.0])
        self.assertIs(payload["use_ddim"], True)

    def test_extract_action_chunk_accepts_common_shapes(self) -> None:
        np.testing.assert_allclose(extract_action_chunk({"actions": np.ones((1, 2, 3))}), np.ones((2, 3)))
        np.testing.assert_allclose(extract_action_chunk({"actions": np.ones((3,))}), np.ones((1, 3)))

        with self.assertRaisesRegex(InferSystemProtocolError, "batch size 1"):
            extract_action_chunk({"actions": np.ones((2, 2, 3))})

    def test_msgpack_safe_converts_numpy_values(self) -> None:
        converted = msgpack_safe({"arr": np.asarray([1, 2]), "scalar": np.float32(1.5)})
        self.assertEqual(converted, {"arr": [1, 2], "scalar": 1.5})


if __name__ == "__main__":
    unittest.main()
