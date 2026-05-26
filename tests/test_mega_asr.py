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
        if len(args) > 0 and args[0] == 1:
            return 4
        return [1, 32]

    def squeeze(self, *args, **kwargs):
        return self

    def transpose(self, *args, **kwargs):
        return self

    def unsqueeze(self, *args, **kwargs):
        return self

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
sys.modules['torch.nn'] = mock_nn
sys.modules['torch.nn.functional'] = mock_nn.functional
sys.modules['librosa'] = MagicMock()
sys.modules['torchaudio'] = MagicMock()
sys.modules['torchaudio.transforms'] = MagicMock()
sys.modules['safetensors'] = MagicMock()

# Mock safetensors.torch load_file
mock_safetensors_torch = MagicMock()
mock_safetensors_torch.load_file = MagicMock()
mock_safetensors_torch.safe_open = MagicMock()

# Mock safe_open context manager
class MockSafeOpen:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    def metadata(self):
        return {"config": '{"model": {"n_mels": 80, "d_model": 192}}'}

mock_safetensors_torch.safe_open = MockSafeOpen
sys.modules['safetensors.torch'] = mock_safetensors_torch

import unittest
from app.services.asr.mega_asr_engine import AudioQualityRouter, LoRADeltaSwitch

class TestMegaASR(unittest.TestCase):
    @patch('app.services.asr.mega_asr_engine.safe_load_file')
    def test_audio_quality_router_load(self, mock_safe_load_file):
        # Mock weights
        mock_weights = {
            "frontend.conv.0.weight": MockTensor(),
            "frontend.conv.0.bias": MockTensor(),
            "classifier.0.weight": MockTensor(),
            "classifier.0.bias": MockTensor()
        }
        mock_safe_load_file.return_value = mock_weights

        with (
            patch('os.path.exists', return_value=True),
            patch('app.services.asr.mega_asr_engine.create_audio_quality_model', return_value=MagicMock()),
            patch('app.services.asr.mega_asr_engine.LogMelSpectrogram', return_value=MagicMock())
        ):
            router = AudioQualityRouter(model_path="dummy_path", device="cpu")
            self.assertEqual(router.device, "cpu")
            self.assertEqual(router.model_path, "dummy_path")

    @patch('app.services.asr.mega_asr_engine.LoRADeltaSwitch._load_adapter_config')
    @patch('app.services.asr.mega_asr_engine.LoRADeltaSwitch._load_adapter_state')
    @patch('app.services.asr.mega_asr_engine.LoRADeltaSwitch._load_adapter_blocks')
    def test_lora_delta_switch_precompute(self, mock_load_blocks, mock_load_state, mock_load_config):
        # Mock configs
        mock_load_config.return_value = {"r": 8, "lora_alpha": 16, "fan_in_fan_out": False}
        mock_load_blocks.return_value = {}

        # Base model mock parameters
        param1 = MockTensor()
        param1.shape = (4, 4)

        base_model = MagicMock()
        base_model.named_modules.return_value = {
            "layer.q_proj": MagicMock(weight=param1)
        }

        # LoRA weights mock
        a = MockTensor()
        b = MockTensor()
        mock_load_state.return_value = {
            "layer.q_proj.lora_A.weight": a,
            "layer.q_proj.lora_B.weight": b
        }

        # Mock matmul to return a MockTensor with correct shape
        expected_delta = MockTensor()
        expected_delta.shape = (4, 4)
        mock_torch.matmul.return_value = expected_delta

        # Initialize switch
        switch = LoRADeltaSwitch(
            base_model=base_model,
            adapter_dir="dummy_dir",
            keep_delta_on_gpu=True
        )

        self.assertEqual(len(switch.items), 1)
        item = switch.items[0]
        self.assertEqual(item["module_name"], "layer.q_proj")
        self.assertEqual(item["weight"].shape, (4, 4))

if __name__ == "__main__":
    sys.exit(unittest.main())
