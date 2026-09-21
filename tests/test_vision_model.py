"""
Tests for EfficientNet-B4 model and Grad-CAM.
Prompt 2.9 — test_vision_model.py
"""

import torch
import numpy as np
import pytest


class TestVisionModel:

    def test_forward_pass_shape(self, vision_model, sample_tensor):
        """Output logits must be (batch, 38)."""
        with torch.no_grad():
            logits = vision_model(sample_tensor)
        assert logits.shape == (1, 38), (
            f"Expected (1, 38) but got {tuple(logits.shape)}"
        )

    def test_confidence_sum(self, vision_model, sample_tensor):
        """Softmax probabilities must sum to 1.0 per sample."""
        with torch.no_grad():
            logits = vision_model(sample_tensor)
            probs  = torch.softmax(logits, dim=1)
        total = probs.sum(dim=1).item()
        assert abs(total - 1.0) < 1e-5, f"Probs sum to {total}, expected 1.0"

    def test_output_dtype(self, vision_model, sample_tensor):
        """Model output must be float32."""
        with torch.no_grad():
            logits = vision_model(sample_tensor)
        assert logits.dtype == torch.float32

    def test_forward_features_shape(self, vision_model, sample_tensor):
        """forward_features must return spatial map (1, C, h, w)."""
        with torch.no_grad():
            feats = vision_model.forward_features(sample_tensor)
        assert feats.ndim == 4, f"Expected 4-D tensor, got {feats.ndim}-D"
        assert feats.shape[0] == 1, "Batch dim should be 1"

    def test_get_target_layer(self, vision_model):
        """get_target_layer must return an nn.Module."""
        import torch.nn as nn
        layer = vision_model.get_target_layer()
        assert isinstance(layer, nn.Module), "Target layer is not an nn.Module"

    def test_gradcam_output_shape(self, vision_model, sample_tensor):
        """Grad-CAM heatmap must match input spatial dimensions (224×224)."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.vision.gradcam import GradCAM

        cam = GradCAM(vision_model, vision_model.get_target_layer())
        heatmap = cam.generate_cam(sample_tensor, target_class=0)
        cam.remove_hooks()

        assert heatmap.ndim  == 2, f"Expected 2-D heatmap, got {heatmap.ndim}-D"
        assert heatmap.dtype == np.float32

    def test_gradcam_range(self, vision_model, sample_tensor):
        """Grad-CAM values must be in [0, 1]."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.vision.gradcam import GradCAM

        cam = GradCAM(vision_model, vision_model.get_target_layer())
        heatmap = cam.generate_cam(sample_tensor, target_class=0)
        cam.remove_hooks()

        assert heatmap.min() >= 0.0, f"Heatmap min below 0: {heatmap.min()}"
        assert heatmap.max() <= 1.0, f"Heatmap max above 1: {heatmap.max()}"

    def test_no_nan_in_output(self, vision_model, sample_tensor):
        """Model output must contain no NaN or Inf values."""
        with torch.no_grad():
            logits = vision_model(sample_tensor)
        assert not torch.isnan(logits).any(), "NaN found in logits"
        assert not torch.isinf(logits).any(), "Inf found in logits"
