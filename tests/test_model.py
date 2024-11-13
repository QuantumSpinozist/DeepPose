import torch
import pytest
import sys
import os


# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.model import DeepPoseRegressor

@pytest.fixture
def model():
    return DeepPoseRegressor()

@pytest.fixture
def input_tensor():
    return torch.randn(1, 3, 220, 220)  # Dummy input tensor

def test_forward_pass_output_shape(model, input_tensor):
    output = model(input_tensor)
    assert output.shape == (1, 14, 2), "Output shape mismatch!"

def test_model_parameters(model):
    assert len(list(model.parameters())) > 0, "Model should have parameters."
