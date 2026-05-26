import sys
from unittest.mock import MagicMock, patch

# Mock torch, librosa, and safetensors.torch before importing the engine
mock_torch = MagicMock()

# Setup torch.no_grad as a context manager
class MockNoGradContext:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

mock_torch.no_grad = MockNoGradContext
mock_torch.float32 = MagicMock()
mock_torch.Size = tuple

# Create a mock tensor class
class MockTensor(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = "cpu"
        self.dtype = "float32"
        self._shape = (4, 4)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, val):
        self._shape = val

    def to(self, *args, **kwargs):
        return self

    def item(self):
        return 0.75  # Return a test value >= 0.5 threshold

    def view(self, *args, **kwargs):
        return self

    def size(self, *args):
        return [1, 32]

    def __mul__(self, other):
        res = MockTensor()
        res.shape = self.shape
        return res

    def __rmul__(self, other):
        res = MockTensor()
        res.shape = self.shape
        return res

mock_torch.Tensor = MockTensor
mock_torch.matmul = MagicMock(return_value=MockTensor())

# Mock torch nn functional elements
mock_nn = MagicMock()
mock_nn.functional.conv2d = MagicMock(return_value=MockTensor())
mock_nn.functional.max_pool2d = MagicMock(return_value=MockTensor())
mock_nn.functional.adaptive_avg_pool2d = MagicMock(return_value=MockTensor())
mock_nn.functional.linear = MagicMock(return_value=MockTensor())
mock_torch.nn = mock_nn

sys.modules['torch'] = mock_torch
sys.modules['librosa'] = MagicMock()
sys.modules['safetensors'] = MagicMock()

# Mock safetensors.torch load_file
mock_safetensors_torch = MagicMock()
mock_safetensors_torch.load_file = MagicMock()
sys.modules['safetensors.torch'] = mock_safetensors_torch

import unittest
from app.services.asr.mega_asr_engine import AudioQualityRouter, LoRADeltaSwitch

class TestMegaASR(unittest.TestCase):
    def test_audio_quality_router_load(self):
        # Mock weights
        mock_weights = {
            "conv1.weight": MockTensor(),
            "conv1.bias": MockTensor(),
            "conv2.weight": MockTensor(),
            "conv2.bias": MockTensor(),
            "fc.weight": MockTensor(),
            "fc.bias": MockTensor()
        }
        mock_safetensors_torch.load_file.return_value = mock_weights

        with patch('os.path.exists', return_value=True):
            router = AudioQualityRouter(model_path="dummy_path", device="cpu")
            self.assertEqual(router.device, "cpu")
            self.assertEqual(router.model_path, "dummy_path")

    def test_lora_delta_switch_precompute(self):
        # Base model mock parameters
        param1 = MockTensor()
        param1.shape = (4, 4)

        base_model = MagicMock()
        base_model.named_parameters.return_value = [
            ("layer.q_proj.weight", param1)
        ]

        # LoRA weights mock
        a = MockTensor()
        b = MockTensor()
        lora_weights = {
            "layer.q_proj.weight.lora_A.weight": a,
            "layer.q_proj.weight.lora_B.weight": b
        }

        # Mock matmul to return a MockTensor with correct shape
        expected_delta = MockTensor()
        expected_delta.shape = (4, 4)
        mock_torch.matmul.return_value = expected_delta

        # Initialize switch
        switch = LoRADeltaSwitch(
            base_model=base_model,
            lora_weights_dict=lora_weights,
            alpha=16,
            rank=2
        )

        self.assertIn("layer.q_proj.weight", switch.delta_weights)
        delta_w = switch.delta_weights["layer.q_proj.weight"]
        self.assertEqual(delta_w.shape, (4, 4))

if __name__ == "__main__":
    sys.exit(unittest.main())
