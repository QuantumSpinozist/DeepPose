import pytest
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

import sys
import os
import deeplake


# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.train_test import box_normalization, l2_loss, calculate_pcp, train, test_pcp
from src.model import DeepPoseRegressor
from src.dataset import LSPDataset

# --- Unit tests for individual functions ---

def test_box_normalization():
    y_tensor = torch.tensor([[150., 150.], [130., 130.]])
    b = [torch.tensor([110., 110.]), 220., 220.]
    
    # Test without inverse
    normalized = box_normalization(y_tensor, b)
    assert normalized.shape == y_tensor.shape, "Output shape mismatch without inverse."
    
    # Test with inverse
    inverse_normalized = box_normalization(normalized, b, inverse=True)
    assert torch.allclose(inverse_normalized, y_tensor, atol=1e-6), "Inverse normalization failed."

def test_l2_loss():
    tensor1 = torch.ones((14, 2))
    tensor2 = torch.zeros((14, 2))
    
    loss = l2_loss(tensor1, tensor2)

    expected_loss = 392**0.5 
    
    assert torch.isclose(loss, torch.tensor(expected_loss), atol=1e-5), f"L2 loss calculation mismatch. Got {loss}, expected {expected_loss}"


def test_calculate_pcp():
    # Simulate ground truth and predicted joints for a batch of 2 samples
    pred_joints = torch.zeros((2, 14, 2))
    gt_joints = torch.zeros((2, 14, 2))
    gt_joints[:, :2] = 1.0  # Mock some distances
    
    pcp = calculate_pcp(pred_joints, gt_joints, threshold=0.5)
    assert isinstance(pcp, dict), "PCP output type mismatch."
    for limb, value in pcp.items():
        assert 0 <= value <= 100, f"Invalid PCP percentage for {limb}."

# --- Integration tests for train and test_pcp functions ---
transform = transforms.Compose([
        transforms.Resize((220, 220)),
        transforms.ToTensor()
    ])





@pytest.fixture
def train_loader():
    train_kwargs = {'batch_size':32}
    train_ds = deeplake.load("hub://activeloop/lsp-extended")
    train_dataset = LSPDataset(train_ds, transform=transform)
    return DataLoader(train_dataset,**train_kwargs)

@pytest.fixture
def test_loader():
    test_kwargs = {'batch_size':32}
    test_ds = deeplake.load("hub://activeloop/lsp-test")
    test_dataset = LSPDataset(test_ds, transform=transform)

    return DataLoader(test_dataset,**test_kwargs)

@pytest.fixture
def args():
    class Args:
        log_interval = 1
        dry_run = True
    return Args()

@pytest.fixture
def model():
    return DeepPoseRegressor()

@pytest.fixture
def device():
    return  torch.device("cpu")

def test_train(args, train_loader):
    model = DeepPoseRegressor()
    device = torch.device("cpu")
    optimizer = optim.AdamW(model.parameters(), lr=0.1 )
    epoch = 1
    train(args, model, device, train_loader, optimizer, epoch)
    # Since we are not asserting here (as train has no return), this is just to ensure no exceptions occur

def test_test_pcp(args, test_loader):
    model = DeepPoseRegressor()
    device = torch.device("cpu")
    threshold = 0.5
    test_pcp(args, model, device, test_loader, threshold)
    # This is to ensure no exceptions occur; PCP scores are printed directly by the function
