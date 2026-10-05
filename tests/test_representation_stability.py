"""Loader regression tests; no cached checkpoints or GPU required."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "solutions/representation-stability/stability_inference.py"
)
spec = importlib.util.spec_from_file_location("stability_inference", MODULE_PATH)
stability = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stability)


class ModelLoadingTests(unittest.TestCase):
    def tearDown(self):
        stability._load_model_cached.cache_clear()
        stability._loaded_model_id = None

    def test_checkpoint_ids_are_not_filtered(self):
        model_ids = (
            "Qwen/Qwen3.5-4B",
            "Skywork/Skywork-OR1-Math-7B",
            "allenai/Olmo-3-7B-Think",
            "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            "openai/gpt-oss-120b",
            "another/checkpoint",
        )
        config = Mock()
        config.get_text_config.return_value = SimpleNamespace(max_position_embeddings=4096)
        model = SimpleNamespace(config=config)
        inputs = {"input_ids": stability.torch.ones((1, 3), dtype=stability.torch.long)}
        classifier = stability.RobustnessClassifier(hidden_size=2)
        with (
            patch.object(stability, "_load_model", return_value=(Mock(), model)) as load,
            patch.object(stability, "_tokenize", return_value=inputs),
            patch.object(stability, "_encode", return_value=stability.torch.tensor([1.0, 0.0])),
        ):
            for model_id in model_ids:
                with self.subTest(model_id=model_id):
                    predictions = stability.predict_robustness(
                        model_id,
                        ["1 + 1?"],
                        classifier,
                    )
                    self.assertEqual(len(predictions), 1)
                    load.assert_called_with(model_id)
            predictions = stability.predict_robustness(
                "qwen3-8b:low",
                ["1 + 1?"],
                classifier,
            )
            self.assertEqual(len(predictions), 1)
            load.assert_called_with(
                "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
            )

    def test_empty_and_blank_inputs_do_not_load(self):
        with patch.object(stability, "_load_model") as load:
            classifier = stability.RobustnessClassifier(hidden_size=2)
            self.assertEqual(
                stability.predict_robustness("any/model", [], classifier),
                [],
            )
            self.assertEqual(
                stability.predict_robustness(
                    "any/model",
                    ["", " "],
                    classifier,
                ),
                [False, False],
            )
            load.assert_not_called()

    def test_cache_is_evicted_before_next_checkpoint_loads(self):
        def load_checkpoint(*args, **kwargs):
            self.assertEqual(stability._load_model_cached.cache_info().currsize, 0)
            return Mock()

        with (
            patch.object(stability.torch.cuda, "is_available", return_value=True),
            patch.object(stability.torch.cuda, "is_bf16_supported", return_value=True),
            patch.object(stability.torch.cuda, "empty_cache") as empty_cache,
            patch.object(stability.AutoTokenizer, "from_pretrained", return_value=Mock()),
            patch.object(stability.AutoModel, "from_pretrained", side_effect=load_checkpoint) as load,
        ):
            first = stability._load_model("first/model")
            self.assertIs(stability._load_model("first/model"), first)
            del first
            stability._load_model("second/model")
            self.assertEqual(load.call_count, 2)
            self.assertEqual(empty_cache.call_count, 2)
            self.assertEqual(load.call_args.kwargs["device_map"], "auto")
            self.assertTrue(load.call_args.kwargs["local_files_only"])

    def test_loading_errors_propagate(self):
        with patch.object(stability, "_load_model", side_effect=OSError("checkpoint not cached")):
            with self.assertRaisesRegex(OSError, "checkpoint not cached"):
                stability.predict_robustness(
                    "missing/model",
                    ["1 + 1?"],
                    stability.RobustnessClassifier(hidden_size=2),
                )


if __name__ == "__main__":
    unittest.main()
