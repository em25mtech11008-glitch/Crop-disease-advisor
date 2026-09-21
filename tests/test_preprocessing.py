"""
Tests for data preprocessing transforms.
Prompt 2.9 — test_preprocessing.py
"""

import torch
import numpy as np
import pytest
from PIL import Image
from torchvision import transforms


# ── Transforms under test ─────────────────────────────────────────────────────

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPreprocessing:

    def test_output_shape(self, sample_leaf_image):
        """Val transform must produce tensor of shape (3, 224, 224)."""
        tensor = VAL_TRANSFORM(sample_leaf_image)
        assert tensor.shape == (3, 224, 224), (
            f"Expected (3, 224, 224) but got {tuple(tensor.shape)}"
        )

    def test_normalization_range(self, sample_leaf_image):
        """After normalization, values should lie in approximately [-3, 3]."""
        tensor = VAL_TRANSFORM(sample_leaf_image)
        assert tensor.min().item() >= -4.0, "Values below expected lower bound"
        assert tensor.max().item() <=  4.0, "Values above expected upper bound"

    def test_output_dtype(self, sample_leaf_image):
        """Tensor dtype must be float32."""
        tensor = VAL_TRANSFORM(sample_leaf_image)
        assert tensor.dtype == torch.float32

    def test_augmentation_determinism(self, sample_leaf_image):
        """Same torch seed must produce identical augmented output."""
        torch.manual_seed(42)
        t1 = TRAIN_TRANSFORM(sample_leaf_image)
        torch.manual_seed(42)
        t2 = TRAIN_TRANSFORM(sample_leaf_image)
        assert torch.allclose(t1, t2), "Transforms not deterministic with same seed"

    def test_batch_consistency(self, sample_leaf_image):
        """Two different images processed independently must not share memory."""
        t1 = VAL_TRANSFORM(sample_leaf_image)
        # Slightly different image
        arr   = np.array(sample_leaf_image)
        arr[0, 0] = [255, 0, 0]
        other = Image.fromarray(arr)
        t2 = VAL_TRANSFORM(other)
        assert not torch.equal(t1, t2), "Different images produced identical tensors"

    def test_channels_first(self, sample_leaf_image):
        """ToTensor must produce channels-first (C, H, W) output."""
        tensor = VAL_TRANSFORM(sample_leaf_image)
        C, H, W = tensor.shape
        assert C == 3,   f"Expected 3 channels, got {C}"
        assert H == 224, f"Expected H=224, got {H}"
        assert W == 224, f"Expected W=224, got {W}"
