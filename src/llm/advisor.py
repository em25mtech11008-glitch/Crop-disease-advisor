"""
CropDiseaseAdvisor — LLaMA 3 8B + QLoRA inference wrapper.
Prompt 2.3 — LLM Inference Class
"""

import json
import argparse
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# ── Constants ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert agricultural advisor with deep knowledge of plant diseases, "
    "organic and chemical treatments, and regional farming practices across India. "
    "Always respond with a valid JSON object following the specified schema."
)

from src.llm.generate_dataset import normalise_disease, DISEASE_ALIAS_MAP


REQUIRED_KEYS = {
    "disease_confirmed", "crop", "severity_assessment",
    "organic_treatments", "chemical_treatments", "preventive_measures",
    "yield_impact_estimate", "urgency", "regional_advisory", "seasonal_advisory",
}


# ── Advisor Class ─────────────────────────────────────────────────────────────

class CropDiseaseAdvisor:
    """
    Wraps fine-tuned LLaMA 3 8B + QLoRA adapter for treatment plan generation.
    """

    def __init__(self, base_model: str, adapter_path: str, device: str = "auto"):
        print(f"[Advisor] Loading base model: {base_model}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map=device,
            trust_remote_code=True,
        )
        print(f"[Advisor] Applying LoRA adapter: {adapter_path}")
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()
        print("[Advisor] Ready ✓")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_prompt(self, disease, crop, region, season, severity) -> str:
        d_pretty = disease.replace("___", " — ").replace("_", " ")
        instruction = (
            f"A farmer in {region} is growing {crop} during the {season} season. "
            f"The crop has been diagnosed with {d_pretty} at a {severity} level. "
            f"Provide a complete integrated disease management plan."
        )
        input_ctx = f"Disease: {d_pretty}\nCrop: {crop}\nRegion: {region}\nSeason: {season}\nSeverity: {severity}"
        
        prompt = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n"
            f"{SYSTEM_PROMPT}\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{instruction}\n\nContext: {input_ctx}\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
        return prompt

    def _generate(self, prompt: str, temperature: float, max_new_tokens: int) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        # Strip prompt tokens
        generated = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    # ── Public Interface ──────────────────────────────────────────────────────

    def validate_output(self, response: dict) -> bool:
        """Check that all required keys exist in the response dict."""
        return REQUIRED_KEYS.issubset(response.keys())

    def generate_treatment_plan(
        self,
        disease:        str,
        crop:           str,
        region:         str,
        season:         str,
        severity:       str,
        max_new_tokens: int = 512,
    ) -> dict:
        """
        Generate a structured treatment plan.

        Returns a Python dict matching the treatment plan schema.
        Retries up to 2 times with lower temperature if JSON parsing fails.
        """
        # Normalise to the shorter name the LLM was fine-tuned on
        disease_norm = normalise_disease(disease)
        prompt = self._build_prompt(disease_norm, crop, region, season, severity)

        temperatures = [0.6, 0.2]
        for attempt, temp in enumerate(temperatures):
            raw = self._generate(prompt, temperature=temp, max_new_tokens=max_new_tokens)

            # ── Robust JSON extraction ────────────────────────────────────
            original_raw = raw
            # 1. Strip markdown code fences (```json ... ``` or ``` ... ```)
            import re
            raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
            raw = raw.replace("```", "").strip()

            # 2. Find the outermost JSON object
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start != -1 and end > start:
                raw = raw[start:end]

            # 3. Remove trailing commas before } or ] (common LLM mistake)
            raw = re.sub(r",\s*([}\]])", r"\1", raw)

            # Debug save logic: always append raw to debug_preds.txt
            try:
                with open("debug_preds.txt", "a", encoding="utf-8") as f:
                    f.write(f"\n--- ATTEMPT {attempt+1} RAW OUTPUT ---\n")
                    f.write(original_raw)
                    f.write("\n--- AFTER PRE-PROCESSING ---\n")
                    f.write(raw)
                    f.write("\n")
            except Exception:
                pass


            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                # If strict JSON fails, try python AST eval (handles single quotes and trailing commas natively)
                try:
                    import ast
                    ast_raw = raw.replace("true", "True").replace("false", "False").replace("null", "None")
                    result = ast.literal_eval(ast_raw)
                except Exception as e:
                    print(f"[Advisor] Attempt {attempt+1}: Both JSON & AST parsing failed.")
                    continue

            if isinstance(result, dict):
                # Map keys from dataset format to API format
                if "urgency" in result and "action_urgency" not in result:
                    result["action_urgency"] = result["urgency"]
                if "regional_advisory" in result and "regional_notes" not in result:
                    result["regional_notes"] = result["regional_advisory"]
                if "seasonal_advisory" in result and "seasonal_notes" not in result:
                    result["seasonal_notes"] = result["seasonal_advisory"]

                if self.validate_output(result):
                    print(f"[Advisor] Success on attempt {attempt+1}")
                    return result
                else:
                    # Patch any missing required keys with empty defaults
                    for k in REQUIRED_KEYS - result.keys():
                        result[k] = [] if k.endswith("treatments") or k == "preventive_measures" else ""
                    print(f"[Advisor] Attempt {attempt+1}: patched missing keys")
                    return result

        # ── DISEASE_DB fallback — real data, not blanks ───────────────────
        print("[Advisor] LLM output parsing failed — using robust curated DISEASE_DB fallback")
        return self._db_fallback(disease_norm, disease, crop, region, season, severity)

    def _db_fallback(self, disease_norm, disease_raw, crop, region, season, severity) -> dict:
        """Build a high-quality response directly from the curated DISEASE_DB."""
        try:
            import sys, pathlib
            sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
            from src.llm.generate_dataset import (
                DISEASE_DB, REGIONAL_NOTES, SEASONAL_NOTES, URGENCY_MAP
            )
            db = DISEASE_DB.get(disease_norm, {})

            severity_long = {
                "Mild":     "Mild (0–30% infection)",
                "Moderate": "Moderate (30–60% infection)",
                "Severe":   "Severe (60–100% infection)",
            }.get(severity, "Moderate (30–60% infection)")

            # Match regional notes to broad region key
            reg_note = next(
                (v for k, v in REGIONAL_NOTES.items() if region.split()[0].lower() in k.lower()),
                "Follow local ICAR recommendations for your region."
            )
            sea_note = next(
                (v for k, v in SEASONAL_NOTES.items() if season.split()[0].lower() in k.lower()),
                "Refer to seasonal crop advisory bulletins."
            )

            organics  = db.get("organic",   [])
            chemicals = db.get("chemical",  [])
            preventive= db.get("preventive",[])

            return {
                "disease_confirmed":   disease_raw,
                "crop":                crop,
                "severity_assessment": severity,
                "organic_treatments":  [
                    {"method": t["method"], "application": t["application"], "frequency": t["frequency"]}
                    for t in organics
                ],
                "chemical_treatments": [
                    {"product": t["product"], "dosage": t["dosage"], "safety_note": t.get("safety_note", t.get("timing",""))}
                    for t in chemicals
                ],
                "preventive_measures": preventive,
                "yield_impact_estimate": db.get("yield_impact", "Variable — consult agronomist."),
                "action_urgency":      URGENCY_MAP.get(severity_long, "Within 3 days"),
                "regional_notes":      reg_note,
                "seasonal_notes":      sea_note,
            }
        except Exception as e:
            print(f"[Advisor] DISEASE_DB fallback failed: {e}")
            return {
                "disease_confirmed":   disease_raw,
                "crop":                crop,
                "severity_assessment": severity,
                "organic_treatments":  [],
                "chemical_treatments": [],
                "preventive_measures": ["Consult your local ICAR agricultural extension officer."],
                "yield_impact_estimate": "Unknown — consult agronomist.",
                "action_urgency":      "Within 3 days",
                "regional_notes":      "See local ICAR recommendations.",
                "seasonal_notes":      "Refer to seasonal crop advisory bulletins.",
            }


# ── CLI Test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",   default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--adapter_path", default="models/llm/llama3_qlora_adapter")
    parser.add_argument("--disease",      default="Tomato___Late_blight")
    parser.add_argument("--crop",         default="Tomato")
    parser.add_argument("--region",       default="North India")
    parser.add_argument("--season",       default="Kharif (Monsoon)")
    parser.add_argument("--severity",     default="Severe (60–100%)")
    args = parser.parse_args()

    advisor = CropDiseaseAdvisor(args.base_model, args.adapter_path)
    plan    = advisor.generate_treatment_plan(
        disease=args.disease, crop=args.crop,
        region=args.region,   season=args.season,
        severity=args.severity,
    )
    print("\n── Treatment Plan ─────────────────────────────────")
    print(json.dumps(plan, indent=2, ensure_ascii=False))
