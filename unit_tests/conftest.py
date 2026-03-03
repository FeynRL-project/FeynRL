import sys
from unittest.mock import MagicMock

# Define mocks for heavy dependencies that might fail
mock_modules = [
    "transformers",
    "transformers.models.auto",
    "transformers.models.auto.modeling_auto",
    "transformers.models.auto.auto_factory",
    "transformers.generation",
    "transformers.generation.utils",
    "deepspeed",
    "deepspeed.zero",
    "peft",
    "safetensors",
    "safetensors.torch"
]

for mod in mock_modules:
    sys.modules[mod] = MagicMock()

# Mock ray correctly with subpackages
class MockRay(MagicMock):
    def remote(self, arg):
        # Handle @ray.remote or ray.remote(class)(params)
        if callable(arg):
             return arg
        return lambda x: x

ray_mock = MockRay()
sys.modules["ray"] = ray_mock
sys.modules["ray.exceptions"] = MagicMock()

# Specifically ensure GenerationMixin exists if it is accessed
import transformers
transformers.generation.GenerationMixin = MagicMock()
transformers.AutoConfig = MagicMock()
transformers.AutoModelForCausalLM = MagicMock()

# Ensure PeftModel is a class for isinstance checks
import peft
class PeftModel: pass
peft.PeftModel = PeftModel
peft.get_peft_model = MagicMock()
peft.LoraConfig = MagicMock()

import torch
import torch.nn as nn
import pytest
import random
import numpy as np

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

@pytest.fixture(autouse=True)
def run_set_seed():
    set_seed(42)

@pytest.fixture
def tiny_model():
    from unit_tests.models import TinyModel
    return TinyModel()

@pytest.fixture
def tiny_value_model():
    from unit_tests.models import TinyValueModel
    return TinyValueModel()
