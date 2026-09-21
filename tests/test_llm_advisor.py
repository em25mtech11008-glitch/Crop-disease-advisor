"""
Tests for the LLM advisor — JSON schema, retry logic, severity mapping.
Prompt 2.9 — test_llm_advisor.py
"""

import json
import pytest
from unittest.mock import MagicMock, patch


# ── Constants ─────────────────────────────────────────────────────────────────

REQUIRED_KEYS = {
    "disease_confirmed", "crop", "severity_assessment",
    "organic_treatments", "chemical_treatments", "preventive_measures",
    "yield_impact_estimate", "action_urgency", "regional_notes", "seasonal_notes",
}

VALID_PLAN = {
    "disease_confirmed":   "Tomato — Late blight",
    "crop":                "Tomato",
    "severity_assessment": "Severe (60–100%)",
    "organic_treatments":  [{"method": "Copper spray", "application": "Foliar", "frequency": "7 days"}],
    "chemical_treatments": [{"product": "Mancozeb", "dosage": "2.5g/L", "timing": "AM", "safety_note": "PHI 7d"}],
    "preventive_measures": ["Crop rotation", "Use resistant varieties"],
    "yield_impact_estimate": "30–50% loss",
    "action_urgency":      "Immediate",
    "regional_notes":      "High humidity in North India promotes late blight.",
    "seasonal_notes":      "Peak risk during Kharif monsoon season.",
}


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestLLMAdvisor:

    def test_output_schema(self):
        """All required keys must be present in a valid treatment plan."""
        assert REQUIRED_KEYS.issubset(VALID_PLAN.keys()), (
            f"Missing keys: {REQUIRED_KEYS - VALID_PLAN.keys()}"
        )

    def test_validate_output_passes(self):
        """validate_output() returns True when all required keys present."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

        # Mock the heavy model loading
        with patch("src.llm.advisor.AutoTokenizer"), \
             patch("src.llm.advisor.AutoModelForCausalLM"), \
             patch("src.llm.advisor.PeftModel"):
            from src.llm.advisor import CropDiseaseAdvisor
            advisor = CropDiseaseAdvisor.__new__(CropDiseaseAdvisor)
            assert advisor.validate_output(VALID_PLAN) is True

    def test_validate_output_fails_on_missing_key(self):
        """validate_output() returns False when key is missing."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

        with patch("src.llm.advisor.AutoTokenizer"), \
             patch("src.llm.advisor.AutoModelForCausalLM"), \
             patch("src.llm.advisor.PeftModel"):
            from src.llm.advisor import CropDiseaseAdvisor
            advisor  = CropDiseaseAdvisor.__new__(CropDiseaseAdvisor)
            bad_plan = {k: v for k, v in VALID_PLAN.items() if k != "action_urgency"}
            assert advisor.validate_output(bad_plan) is False

    def test_json_retry_on_parse_error(self):
        """If JSON is malformed on first attempt, advisor retries and returns fallback."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

        with patch("src.llm.advisor.AutoTokenizer"), \
             patch("src.llm.advisor.AutoModelForCausalLM"), \
             patch("src.llm.advisor.PeftModel"):
            from src.llm.advisor import CropDiseaseAdvisor
            advisor  = CropDiseaseAdvisor.__new__(CropDiseaseAdvisor)
            # All _generate calls return invalid JSON
            advisor._generate = MagicMock(return_value="this is not json {{{}}")
            result = advisor.generate_treatment_plan(
                disease="Tomato___Late_blight", crop="Tomato",
                region="North India", season="Kharif (Monsoon)",
                severity="Severe (60–100%)",
            )
            # Must return a valid fallback dict with all required keys
            assert REQUIRED_KEYS.issubset(result.keys()), (
                f"Fallback missing keys: {REQUIRED_KEYS - result.keys()}"
            )
            # _generate should have been called 3 times (initial + 2 retries)
            assert advisor._generate.call_count == 3

    def test_severity_mapping(self):
        """Confidence → severity rules must be respected."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.pipeline import confidence_to_severity

        assert confidence_to_severity(0.90) == "Severe (60–100%)"
        assert confidence_to_severity(0.65) == "Moderate (30–60%)"
        assert confidence_to_severity(0.30) == "Mild (0–30%)"
        # Boundary: exactly 0.80 → Severe
        assert confidence_to_severity(0.80) == "Severe (60–100%)"
        # Boundary: exactly 0.50 → Moderate
        assert confidence_to_severity(0.50) == "Moderate (30–60%)"

    def test_urgency_is_valid(self):
        """action_urgency in VALID_PLAN must be one of the allowed values."""
        valid_urgencies = {"Immediate", "Within 3 days", "Within a week", "Monitor"}
        assert VALID_PLAN["action_urgency"] in valid_urgencies
