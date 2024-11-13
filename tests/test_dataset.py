import torch
import pytest

from torchvision import transforms
import deeplake

import sys
import os


# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.dataset import LSPDataset

@pytest.fixture
def setup_dataset():
    # Mocking a simple Deeplake dataset for testing
    ds = deeplake.load("hub://activeloop/lsp-train")  # Use a valid dataset link
    transform = transforms.Compose([transforms.Resize((220, 220)), transforms.ToTensor()])
    dataset = LSPDataset(ds, transform=transform)
    return ds, dataset

def test_length(setup_dataset):
    ds, dataset = setup_dataset
    assert len(dataset) == len(ds), "Length of dataset does not match."

def test_get_item(setup_dataset):
    _, dataset = setup_dataset
    image, label = dataset[0]
    assert image.shape == (3, 220, 220), "Image shape mismatch."
    assert label.shape == (14, 2), "Label shape mismatch."
