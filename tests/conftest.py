"""
pytest conftest — shared fixtures for all test modules.
"""

import pytest
import torch
import numpy as np
from PIL import Image


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_leaf_image() -> Image.Image:
    """Synthetic 224×224 green PIL image simulating a leaf photo."""
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[:, :, 1] = 120   # green channel dominant
    arr[:, :, 0] = 34
    arr[:, :, 2] = 34
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture(scope="session")
def sample_tensor(sample_leaf_image) -> torch.Tensor:
    """Normalised (1, 3, 224, 224) tensor from the leaf image."""
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(sample_leaf_image).unsqueeze(0)  # (1, 3, 224, 224)


@pytest.fixture(scope="session")
def class_names():
    """38 dummy class names matching PlantVillage structure."""
    import sys
    from pathlib import Path
    import json
    names_path = Path("data/processed/class_names.json")
    if names_path.exists():
        with open(names_path) as f:
            mapping = json.load(f)
            return [mapping[str(i)] for i in range(len(mapping))]
    # Fallback: synthetic names
    return [f"Crop_{i}___Disease_{i}" for i in range(38)]


@pytest.fixture(scope="session")
def vision_model(class_names):
    """Load EfficientNet-B4 with random weights (no checkpoint needed for unit tests)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.vision.model import EfficientNetB4Classifier
    model = EfficientNetB4Classifier(num_classes=len(class_names), dropout=0.3)
    model.eval()
    return model
