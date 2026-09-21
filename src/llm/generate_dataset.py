"""
Synthetic instruction dataset generator for LLaMA 3 QLoRA fine-tuning.
Expanded version: all 38 PlantVillage classes, 12 instruction templates,
15 Indian regions, 3 farmer profiles, default 50 000 pairs.
"""

import json, random, argparse
from pathlib import Path
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# Disease Normalization (Moved from advisor.py to prevent heavy LLM imports)
# ─────────────────────────────────────────────────────────────────────────────
DISEASE_ALIAS_MAP = {
    # Corn / Maize
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Corn___Cercospora_leaf_spot",
    "Corn_(maize)___Common_rust_":                        "Corn___Common_rust",
    "Corn_(maize)___Northern_Leaf_Blight":                "Corn___Northern_Leaf_Blight",
    "Corn_(maize)___healthy":                             "Corn___healthy",
    # Pepper
    "Pepper,_bell___Bacterial_spot":                      "Pepper___Bacterial_spot",
    "Pepper,_bell___healthy":                             "Pepper___healthy",
    # Tomato
    "Tomato___Spider_mites Two-spotted_spider_mite":      "Tomato___Spider_mites",
    # Grape
    "Grape___Esca_(Black_Measles)":                       "Grape___Esca",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":         "Grape___Leaf_blight",
    # Orange
    "Orange___Haunglongbing_(Citrus_greening)":           "Orange___Haunglongbing",
    # Cherry
    "Cherry_(including_sour)___Powdery_mildew":           "Cherry___Powdery_mildew",
    "Cherry_(including_sour)___healthy":                  "Cherry___healthy",
}

def normalise_disease(raw_label: str) -> str:
    """Convert a full PlantVillage folder name to the LLM-training canonical name."""
    return DISEASE_ALIAS_MAP.get(raw_label, raw_label)


# ─────────────────────────────────────────────────────────────────────────────
# Complete Disease Knowledge Base  (all 38 PlantVillage classes)
# ─────────────────────────────────────────────────────────────────────────────
DISEASE_DB = {
    # ── TOMATO ───────────────────────────────────────────────────────────────
    "Tomato___Late_blight": {
        "crop": "Tomato", "severity_modifier": 1.4,
        "organic": [
            {"method": "Copper-based fungicide",      "application": "Foliar spray, both leaf surfaces", "frequency": "Every 7 days"},
            {"method": "Neem oil (5 ml/L)",           "application": "Thorough foliar spray",            "frequency": "Every 5 days"},
            {"method": "Baking soda solution (1%)",   "application": "Spray at first symptom",           "frequency": "Every 3 days"},
        ],
        "chemical": [
            {"product": "Mancozeb 75% WP",            "dosage": "2.5 g/L", "timing": "Preventive; every 7 days", "safety_note": "PHI 7 days; gloves required"},
            {"product": "Metalaxyl + Mancozeb",       "dosage": "2 g/L",   "timing": "Before rain forecast",      "safety_note": "PHI 14 days; avoid inhalation"},
            {"product": "Cymoxanil 8% + Famoxadone",  "dosage": "2 g/L",   "timing": "At disease onset",          "safety_note": "PHI 3 days"},
        ],
        "preventive": [
            "Use certified disease-free seeds / resistant varieties (e.g., Mountain Magic)",
            "Maintain 60–70 cm row spacing for air circulation",
            "Switch to drip irrigation; avoid overhead watering",
            "Remove and destroy infected debris immediately",
            "Rotate crops — avoid all Solanaceae for 2 seasons",
            "Monitor weather; spray before forecasted rain",
        ],
        "yield_impact": "30–50% yield loss if untreated; up to 100% in wet seasons",
    },
    "Tomato___Early_blight": {
        "crop": "Tomato", "severity_modifier": 1.1,
        "organic": [
            {"method": "Copper oxychloride",  "application": "Cover leaf undersides",               "frequency": "Every 10 days"},
            {"method": "Compost tea (1:5)",   "application": "Foliar; strengthens plant immunity",  "frequency": "Weekly"},
            {"method": "Garlic extract spray","application": "15 g crushed garlic / L",             "frequency": "Every 5 days"},
        ],
        "chemical": [
            {"product": "Chlorothalonil 75% WP", "dosage": "2 g/L",   "timing": "Preventive, before onset", "safety_note": "PHI 7 days; wear mask"},
            {"product": "Azoxystrobin",          "dosage": "1 ml/L",  "timing": "At first symptom",         "safety_note": "PHI 3 days"},
            {"product": "Iprodione 50% WP",      "dosage": "2 g/L",   "timing": "After rainfall",           "safety_note": "PHI 7 days"},
        ],
        "preventive": [
            "Remove lower leaves in contact with soil",
            "Mulch to prevent soil splash",
            "Adequate potassium; avoid excess nitrogen",
            "Stake plants for better air flow",
        ],
        "yield_impact": "20–40% reduction; fruit quality also affected",
    },
    "Tomato___Bacterial_spot": {
        "crop": "Tomato", "severity_modifier": 1.0,
        "organic": [
            {"method": "Copper hydroxide",   "application": "Foliar spray",      "frequency": "Every 7 days"},
            {"method": "Hydrogen peroxide 3%","application": "Dilute 1:4, foliar","frequency": "Every 5 days"},
        ],
        "chemical": [
            {"product": "Streptomycin sulfate (200 ppm)", "dosage": "Per label", "timing": "At onset",     "safety_note": "Bactericide; PHI 1 day"},
            {"product": "Copper-based bactericide",       "dosage": "2.5 g/L",   "timing": "Preventive",   "safety_note": "PHI 0 days"},
        ],
        "preventive": [
            "Use resistant varieties",
            "Avoid working in wet field conditions",
            "Sanitize tools with 10% bleach solution",
            "No overhead irrigation",
        ],
        "yield_impact": "10–25% loss in marketable fruit",
    },
    "Tomato___Leaf_Mold": {
        "crop": "Tomato", "severity_modifier": 0.9,
        "organic": [
            {"method": "Neem oil",            "application": "Foliar, especially leaf undersides", "frequency": "Every 7 days"},
            {"method": "Potassium bicarbonate","application": "5 g/L water, foliar",               "frequency": "Every 5 days"},
        ],
        "chemical": [
            {"product": "Chlorothalonil",  "dosage": "2 g/L",   "timing": "Preventive",    "safety_note": "PHI 7 days"},
            {"product": "Mandipropamid",   "dosage": "0.5 ml/L","timing": "At first sign", "safety_note": "PHI 1 day"},
        ],
        "preventive": [
            "Ensure greenhouse ventilation above 85% RH",
            "Remove infected leaves promptly",
            "Plant at recommended spacing",
        ],
        "yield_impact": "15–30% loss in protected cultivation",
    },
    "Tomato___Septoria_leaf_spot": {
        "crop": "Tomato", "severity_modifier": 1.0,
        "organic": [
            {"method": "Copper oxychloride", "application": "Foliar spray",    "frequency": "Every 7 days"},
            {"method": "Compost tea",        "application": "Diluted 1:5 foliar","frequency": "Weekly"},
        ],
        "chemical": [
            {"product": "Mancozeb 75% WP",  "dosage": "2 g/L",  "timing": "At first symptom", "safety_note": "PHI 7 days"},
            {"product": "Chlorothalonil",    "dosage": "2 g/L",  "timing": "Preventive",        "safety_note": "PHI 7 days"},
        ],
        "preventive": [
            "Remove infected lower leaves",
            "Avoid wetting foliage",
            "Crop rotation — minimum 2 years",
        ],
        "yield_impact": "20–35% if defoliation is severe",
    },
    "Tomato___Spider_mites": {
        "crop": "Tomato", "severity_modifier": 0.8,
        "organic": [
            {"method": "Neem oil spray",         "application": "Foliar, both sides", "frequency": "Every 3–5 days"},
            {"method": "Strong water jet",       "application": "Dislodge mites",     "frequency": "Daily"},
            {"method": "Predatory mites (Phytoseiidae)", "application": "Biological control", "frequency": "Once"},
        ],
        "chemical": [
            {"product": "Abamectin 1.8% EC", "dosage": "0.75 ml/L", "timing": "At first infestation", "safety_note": "PHI 7 days; highly toxic to bees"},
            {"product": "Spiromesifen",       "dosage": "1 ml/L",    "timing": "Severe infestations",   "safety_note": "PHI 3 days"},
        ],
        "preventive": [
            "Monitor underside of leaves weekly with hand lens",
            "Avoid water stress (stressed plants are more susceptible)",
            "Remove dusty conditions — mites thrive in dry dust",
        ],
        "yield_impact": "10–40% depending on duration of infestation",
    },
    "Tomato___Target_Spot": {
        "crop": "Tomato", "severity_modifier": 1.0,
        "organic": [
            {"method": "Neem oil",       "application": "Foliar spray", "frequency": "Every 7 days"},
            {"method": "Copper soap spray","application": "Full coverage","frequency": "Every 10 days"},
        ],
        "chemical": [
            {"product": "Boscalid + Pyraclostrobin", "dosage": "1 ml/L", "timing": "At first symptom", "safety_note": "PHI 0 days"},
            {"product": "Azoxystrobin",              "dosage": "1 ml/L", "timing": "Preventive",        "safety_note": "PHI 3 days"},
        ],
        "preventive": [
            "Stake and prune for canopy airflow",
            "Avoid high humidity conditions",
            "Rotate crops and destroy debris",
        ],
        "yield_impact": "15–30% loss if unmanaged",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "crop": "Tomato", "severity_modifier": 1.5,
        "organic": [
            {"method": "Yellow sticky traps",       "application": "40 traps/ha to monitor whitefly",  "frequency": "Replace every 2 weeks"},
            {"method": "Reflective mulch",          "application": "Silver plastic mulch on beds",      "frequency": "Season-long"},
            {"method": "Neem oil (whitefly control)","application": "Foliar spray on undersides",       "frequency": "Every 5 days"},
        ],
        "chemical": [
            {"product": "Imidacloprid 17.8% SL",  "dosage": "0.5 ml/L", "timing": "Early; control whitefly vector", "safety_note": "PHI 7 days; toxic to bees"},
            {"product": "Thiamethoxam 25% WG",    "dosage": "0.3 g/L",  "timing": "Preventive spray",               "safety_note": "PHI 3 days"},
        ],
        "preventive": [
            "Plant virus-resistant varieties (e.g., Naveen, Arka Rakshak)",
            "Install fine mesh insect-proof nets in seedling nursery",
            "Rogue out infected plants immediately",
            "Maintain 60-day whitefly-free buffer period before transplanting",
        ],
        "yield_impact": "Up to 100% loss in highly susceptible varieties",
    },
    "Tomato___Tomato_mosaic_virus": {
        "crop": "Tomato", "severity_modifier": 1.3,
        "organic": [
            {"method": "Skim milk spray (10%)", "application": "Coat hands and tools before handling", "frequency": "Before each operation"},
            {"method": "Rogue infected plants",  "application": "Remove and destroy immediately",       "frequency": "Weekly scouting"},
        ],
        "chemical": [
            {"product": "No curative chemical exists", "dosage": "N/A", "timing": "Prevention only", "safety_note": "Focus on vector control and sanitation"},
        ],
        "preventive": [
            "Use virus-indexed certified seed",
            "Wash hands with soap before and after handling plants",
            "Disinfect tools with 10% trisodium phosphate",
            "Control aphid vectors with systemic insecticides",
            "Avoid tobacco use near plants (ToMV related to TMV)",
        ],
        "yield_impact": "20–80% loss depending on infection timing",
    },
    "Tomato___healthy": {
        "crop": "Tomato", "severity_modifier": 0.0,
        "organic":   [],
        "chemical":  [],
        "preventive": [
            "Continue current IPM program",
            "Monitor weekly with 10x hand lens",
            "Maintain soil health with compost",
            "Ensure balanced NPK nutrition",
        ],
        "yield_impact": "No yield impact — plant is healthy",
    },
    # ── APPLE ────────────────────────────────────────────────────────────────
    "Apple___Apple_scab": {
        "crop": "Apple", "severity_modifier": 1.2,
        "organic": [
            {"method": "Bordeaux mixture (1%)", "application": "Before bud break",    "frequency": "Every 10 days"},
            {"method": "Sulfur (wettable)",     "application": "Foliar; after petal fall","frequency": "Every 7–14 days"},
            {"method": "Lime sulfur",           "application": "Dormant spray",        "frequency": "Once pre-season"},
        ],
        "chemical": [
            {"product": "Captan 50% WP",    "dosage": "2.5 g/L", "timing": "Pink bud stage",       "safety_note": "PHI 0 days; avoid inhalation"},
            {"product": "Myclobutanil",      "dosage": "1 ml/L",  "timing": "During leaf wetness",  "safety_note": "PHI 7 days"},
            {"product": "Trifloxystrobin",   "dosage": "0.5 g/L", "timing": "At petal fall",        "safety_note": "PHI 14 days"},
        ],
        "preventive": [
            "Rake and destroy fallen leaves after harvest",
            "Prune canopy for air circulation",
            "Plant resistant varieties: Liberty, Enterprise, GoldRush",
            "Apply urea (5%) to fallen leaves to speed decomposition",
        ],
        "yield_impact": "Up to 70% loss in severe outbreak years",
    },
    "Apple___Black_rot": {
        "crop": "Apple", "severity_modifier": 1.1,
        "organic": [
            {"method": "Lime sulfur",  "application": "Dormant spray pre-bud break", "frequency": "Once pre-season"},
            {"method": "Copper spray", "application": "Foliar at petal fall",         "frequency": "Every 7 days"},
        ],
        "chemical": [
            {"product": "Thiophanate-methyl", "dosage": "1 g/L",  "timing": "Bloom and post-bloom", "safety_note": "PHI 1 day"},
            {"product": "Captan 50% WP",      "dosage": "2 g/L",  "timing": "Before rain events",   "safety_note": "PHI 0 days"},
        ],
        "preventive": [
            "Remove mummified fruit from trees and ground",
            "Prune dead/diseased wood; seal cuts with wound paint",
            "Balanced fertilization; avoid excessive nitrogen",
            "Ensure good drainage to reduce waterlogging",
        ],
        "yield_impact": "15–30% crop loss with fruit quality degradation",
    },
    "Apple___Cedar_apple_rust": {
        "crop": "Apple", "severity_modifier": 1.0,
        "organic": [
            {"method": "Sulfur dust",      "application": "At orange gall stage on cedar", "frequency": "Every 7 days"},
            {"method": "Neem oil",         "application": "Foliar spray on young leaves",  "frequency": "Every 10 days"},
        ],
        "chemical": [
            {"product": "Myclobutanil",  "dosage": "1 ml/L", "timing": "At pink bud to petal fall", "safety_note": "PHI 7 days"},
            {"product": "Propiconazole", "dosage": "1 ml/L", "timing": "Every 10 days spring",       "safety_note": "PHI 14 days"},
        ],
        "preventive": [
            "Remove nearby Eastern red cedars (alternate host) within 1 km if possible",
            "Plant rust-resistant apple varieties",
            "Apply protective fungicide sprays in spring",
        ],
        "yield_impact": "10–25% loss; mainly aesthetic fruit damage",
    },
    "Apple___healthy": {
        "crop": "Apple", "severity_modifier": 0.0,
        "organic": [],
        "chemical": [],
        "preventive": [
            "Continue scheduled dormant spray program",
            "Monitor for scab and fire blight during wet springs",
            "Maintain proper tree nutrition and pruning",
        ],
        "yield_impact": "No yield impact — tree is healthy",
    },
    # ── CORN / MAIZE ─────────────────────────────────────────────────────────
    "Corn___Cercospora_leaf_spot": {
        "crop": "Corn", "severity_modifier": 1.0,
        "organic": [
            {"method": "Trichoderma harzianum bio-fungicide", "application": "Foliar spray", "frequency": "Every 14 days"},
            {"method": "Neem cake soil application",          "application": "250 kg/ha",    "frequency": "Before sowing"},
        ],
        "chemical": [
            {"product": "Azoxystrobin",             "dosage": "1 ml/L",   "timing": "At V6 stage",         "safety_note": "PHI 7 days"},
            {"product": "Propiconazole",             "dosage": "1 ml/L",   "timing": "At first symptoms",   "safety_note": "PHI 30 days"},
            {"product": "Pyraclostrobin + Boscalid","dosage": "0.75 ml/L","timing": "V8 to tasseling",      "safety_note": "PHI 7 days"},
        ],
        "preventive": [
            "Plant resistant hybrids",
            "Adopt 2-year crop rotation",
            "Manage crop residues — plough under or burn",
        ],
        "yield_impact": "10–20% loss; higher if ear leaves are infected",
    },
    "Corn___Common_rust": {
        "crop": "Corn", "severity_modifier": 0.9,
        "organic": [
            {"method": "Neem-based spray",    "application": "Foliar at V6 stage", "frequency": "Every 14 days"},
            {"method": "Sulfur dust",         "application": "Dust on leaves",     "frequency": "Every 10 days"},
        ],
        "chemical": [
            {"product": "Propiconazole",               "dosage": "1 ml/L",   "timing": "At first pustule", "safety_note": "PHI 30 days"},
            {"product": "Tebuconazole",                "dosage": "1 ml/L",   "timing": "Tasseling stage",  "safety_note": "PHI 14 days"},
            {"product": "Azoxystrobin + Propiconazole","dosage": "1.5 ml/L","timing": "V8-R1 stage",       "safety_note": "PHI 7 days"},
        ],
        "preventive": [
            "Plant rust-resistant hybrids (HQPM-1, DHM-117)",
            "Early sowing to avoid peak humidity period",
            "Balanced NPK; avoid excess nitrogen",
        ],
        "yield_impact": "5–15% loss; higher in susceptible varieties under heavy dew",
    },
    "Corn___Northern_Leaf_Blight": {
        "crop": "Corn", "severity_modifier": 1.1,
        "organic": [
            {"method": "Trichoderma bio-fungicide", "application": "Soil drench + foliar", "frequency": "At 30 and 60 DAS"},
            {"method": "Pseudomonas fluorescens",   "application": "Seed treatment",        "frequency": "At sowing"},
        ],
        "chemical": [
            {"product": "Azoxystrobin + Propiconazole", "dosage": "1.5 ml/L","timing": "V8–V10 stage",    "safety_note": "PHI 7 days"},
            {"product": "Mancozeb 75% WP",              "dosage": "2g/L",    "timing": "At V6 preventive","safety_note": "PHI 7 days"},
        ],
        "preventive": [
            "Use resistant hybrids (carries Ht gene)",
            "Crop rotation with legumes",
            "Deep plough crop debris after harvest",
            "Avoid excessive nitrogen fertilization",
        ],
        "yield_impact": "15–30% yield loss in humid regions",
    },
    "Corn___healthy": {
        "crop": "Corn", "severity_modifier": 0.0,
        "organic": [],
        "chemical": [],
        "preventive": [
            "Monitor weekly for early rust or NLB symptoms",
            "Maintain adequate soil moisture for even growth",
            "Side-dress with nitrogen at V6 stage",
        ],
        "yield_impact": "No yield impact — crop is healthy",
    },
    # ── GRAPE ────────────────────────────────────────────────────────────────
    "Grape___Black_rot": {
        "crop": "Grape", "severity_modifier": 1.4,
        "organic": [
            {"method": "Sulfur-based spray", "application": "Cover berries and foliage", "frequency": "Every 7–10 days"},
            {"method": "Copper (0.5%)",      "application": "Preventive foliar",          "frequency": "Pre-bloom"},
        ],
        "chemical": [
            {"product": "Mancozeb",      "dosage": "2 g/L",   "timing": "Pre-bloom to fruit set", "safety_note": "PHI 66 days"},
            {"product": "Myclobutanil",  "dosage": "1 ml/L",  "timing": "At bloom",               "safety_note": "PHI 0 days"},
            {"product": "Ziram",         "dosage": "2 g/L",   "timing": "Pre-bloom",              "safety_note": "PHI 21 days"},
        ],
        "preventive": [
            "Remove infected berries and mummies",
            "Canopy management — leaf removal for air flow",
            "Avoid wetting foliage during irrigation",
            "Time sprays for post-rain application",
        ],
        "yield_impact": "Up to 80% berry loss in wet years",
    },
    "Grape___Esca": {
        "crop": "Grape", "severity_modifier": 1.5,
        "organic": [
            {"method": "Trichoderma-based paste on pruning wounds", "application": "Seal every cut", "frequency": "After each pruning"},
            {"method": "Bordeaux paste (thick)",                    "application": "Paint cut surface","frequency": "Post-pruning"},
        ],
        "chemical": [
            {"product": "Sodium arsenite (restricted)", "dosage": "Per label", "timing": "Trunk injection", "safety_note": "Highly toxic; check local regulations"},
            {"product": "Tebuconazole",                 "dosage": "1 ml/L",    "timing": "Preventive foliar","safety_note": "PHI 14 days"},
        ],
        "preventive": [
            "Prune during dry weather",
            "Apply wound sealants immediately after pruning",
            "Remove and burn severely infected vines",
            "Avoid large pruning wounds",
        ],
        "yield_impact": "Vine decline; 30–60% yield loss over successive years",
    },
    "Grape___Leaf_blight": {
        "crop": "Grape", "severity_modifier": 1.2,
        "organic": [
            {"method": "Copper oxychloride", "application": "Foliar spray", "frequency": "Every 7 days"},
            {"method": "Neem oil",           "application": "Foliar spray", "frequency": "Every 10 days"},
        ],
        "chemical": [
            {"product": "Metalaxyl + Mancozeb", "dosage": "2.5 g/L", "timing": "At first symptoms", "safety_note": "PHI 14 days"},
            {"product": "Captan 50% WP",        "dosage": "2 g/L",   "timing": "Preventive",         "safety_note": "PHI 21 days"},
        ],
        "preventive": [
            "Avoid overhead irrigation",
            "Manage canopy; remove excess shoot growth",
            "Monitor after periods of high humidity",
        ],
        "yield_impact": "15–35% loss; also reduces photosynthesis long-term",
    },
    "Grape___healthy": {
        "crop": "Grape", "severity_modifier": 0.0,
        "organic": [], "chemical": [],
        "preventive": [
            "Continue scheduled dormant spray",
            "Ensure balanced nutrition (K, Mg, Zn)",
            "Maintain vine training system for airflow",
        ],
        "yield_impact": "No yield impact — vines are healthy",
    },
    # ── POTATO ───────────────────────────────────────────────────────────────
    "Potato___Early_blight": {
        "crop": "Potato", "severity_modifier": 1.0,
        "organic": [
            {"method": "Copper oxychloride",   "application": "Foliar",       "frequency": "Every 10 days"},
            {"method": "Neem oil + soap (1%)", "application": "Both leaf sides","frequency": "Every 7 days"},
        ],
        "chemical": [
            {"product": "Chlorothalonil 75% WP", "dosage": "2 g/L",   "timing": "Preventive",     "safety_note": "PHI 7 days"},
            {"product": "Mancozeb 75% WP",       "dosage": "2.5 g/L", "timing": "At first lesion","safety_note": "PHI 7 days"},
        ],
        "preventive": [
            "Maintain adequate soil calcium and potassium",
            "Avoid water stress — maintain even soil moisture",
            "Remove infected plant material",
        ],
        "yield_impact": "15–30% reduction in tuber quality and weight",
    },
    "Potato___Late_blight": {
        "crop": "Potato", "severity_modifier": 1.6,
        "organic": [
            {"method": "Copper hydroxide",  "application": "Foliar; all leaf surfaces", "frequency": "Every 7 days"},
            {"method": "Bordeaux mixture",  "application": "Preventive foliar",          "frequency": "Every 10 days"},
        ],
        "chemical": [
            {"product": "Cymoxanil + Famoxadone", "dosage": "2 g/L",   "timing": "At onset / before rain",  "safety_note": "PHI 3 days"},
            {"product": "Metalaxyl + Mancozeb",   "dosage": "2.5 g/L", "timing": "Preventive; every 7 days","safety_note": "PHI 14 days"},
            {"product": "Fluopicolide",            "dosage": "1 ml/L",  "timing": "Systemic; at first sign", "safety_note": "PHI 7 days"},
        ],
        "preventive": [
            "Use certified seed potatoes free from late blight",
            "Hill up soil around stems to cover tubers",
            "Destroy volunteer potato plants",
            "Avoid irrigation in the evening",
        ],
        "yield_impact": "Complete crop failure possible in wet seasons",
    },
    "Potato___healthy": {
        "crop": "Potato", "severity_modifier": 0.0,
        "organic": [], "chemical": [],
        "preventive": [
            "Monitor weekly for late blight during humid weather",
            "Maintain soil moisture through drip or furrow irrigation",
            "Ensure good drainage to prevent waterlogging",
        ],
        "yield_impact": "No yield impact — crop is healthy",
    },
    # ── PEPPER ───────────────────────────────────────────────────────────────
    "Pepper___Bacterial_spot": {
        "crop": "Pepper", "severity_modifier": 1.1,
        "organic": [
            {"method": "Copper-based spray",   "application": "Foliar; complete coverage", "frequency": "Every 7 days"},
            {"method": "Copper soap liquid",   "application": "Foliar",                    "frequency": "Every 5 days"},
        ],
        "chemical": [
            {"product": "Copper hydroxide",      "dosage": "2.5 g/L", "timing": "Preventive; weekly", "safety_note": "PHI 0 days"},
            {"product": "Streptomycin + Copper", "dosage": "2 g/L",   "timing": "At first lesion",    "safety_note": "PHI 1 day"},
        ],
        "preventive": [
            "Use disease-free transplants from certified nursery",
            "Avoid overhead irrigation",
            "Rotate crops with cereals for 2 years",
            "Sanitize stakes and equipment",
        ],
        "yield_impact": "20–30% loss; fruit lesions render produce unmarketable",
    },
    "Pepper___healthy": {
        "crop": "Pepper", "severity_modifier": 0.0,
        "organic": [], "chemical": [],
        "preventive": [
            "Maintain consistent soil moisture",
            "Monitor for bacterial spot lesions during wet periods",
            "Fertilize with balanced NPK + micronutrients",
        ],
        "yield_impact": "No yield impact — plant is healthy",
    },
    # ── STRAWBERRY ───────────────────────────────────────────────────────────
    "Strawberry___Leaf_scorch": {
        "crop": "Strawberry", "severity_modifier": 1.0,
        "organic": [
            {"method": "Copper oxychloride", "application": "Foliar spray", "frequency": "Every 10 days"},
            {"method": "Sulfur (wettable)",  "application": "Preventive",   "frequency": "Every 7 days"},
        ],
        "chemical": [
            {"product": "Myclobutanil",  "dosage": "1 ml/L",  "timing": "At first symptom", "safety_note": "PHI 0 days"},
            {"product": "Captan 50% WP", "dosage": "2 g/L",   "timing": "Preventive",        "safety_note": "PHI 7 days"},
        ],
        "preventive": [
            "Use certified disease-free runners",
            "Remove and destroy infected leaves",
            "Ensure adequate spacing (30 cm between plants)",
        ],
        "yield_impact": "15–25% loss from reduced photosynthetic area",
    },
    "Strawberry___healthy": {
        "crop": "Strawberry", "severity_modifier": 0.0,
        "organic": [], "chemical": [],
        "preventive": [
            "Use certified runners each season",
            "Monitor for grey mold during fruiting",
            "Mulch with straw to prevent soil splash",
        ],
        "yield_impact": "No yield impact — crop is healthy",
    },
    # ── PEACH ────────────────────────────────────────────────────────────────
    "Peach___Bacterial_spot": {
        "crop": "Peach", "severity_modifier": 1.2,
        "organic": [
            {"method": "Copper hydroxide", "application": "Dormant spray + early season", "frequency": "Every 10 days"},
        ],
        "chemical": [
            {"product": "Oxytetracycline",  "dosage": "200 ppm", "timing": "Petal fall + 2 more",    "safety_note": "PHI 1 day"},
            {"product": "Copper octanoate", "dosage": "1 ml/L",  "timing": "Preventive spring spray", "safety_note": "PHI 0 days"},
        ],
        "preventive": [
            "Plant in sites with good air circulation",
            "Avoid high-nitrogen fertilization late in season",
            "Choose resistant varieties (e.g., Redhaven)",
        ],
        "yield_impact": "15–40% loss; severe lesions cause premature defoliation",
    },
    "Peach___healthy": {
        "crop": "Peach", "severity_modifier": 0.0,
        "organic": [], "chemical": [],
        "preventive": [
            "Schedule dormant copper spray before bud break",
            "Monitor for bacterial spot during wet springs",
            "Maintain balanced soil pH (6.0–6.5)",
        ],
        "yield_impact": "No yield impact — tree is healthy",
    },
    # ── CHERRY ───────────────────────────────────────────────────────────────
    "Cherry___Powdery_mildew": {
        "crop": "Cherry", "severity_modifier": 1.0,
        "organic": [
            {"method": "Potassium bicarbonate (5 g/L)", "application": "Foliar spray", "frequency": "Every 7 days"},
            {"method": "Neem oil",                       "application": "Foliar spray", "frequency": "Every 10 days"},
            {"method": "Milk spray (10%)",               "application": "Foliar",       "frequency": "Every 5 days"},
        ],
        "chemical": [
            {"product": "Myclobutanil",  "dosage": "1 ml/L",   "timing": "At first white powdery spots", "safety_note": "PHI 7 days"},
            {"product": "Sulfur (80%)",  "dosage": "2.5 g/L",  "timing": "Preventive (not above 35°C)",  "safety_note": "PHI 0 days"},
            {"product": "Trifloxystrobin","dosage": "0.5 ml/L","timing": "At early shoot development",    "safety_note": "PHI 7 days"},
        ],
        "preventive": [
            "Avoid excessive nitrogen which promotes soft leafy growth",
            "Prune for canopy openness",
            "Do not apply sulfur-based products when temperature > 35°C",
        ],
        "yield_impact": "10–30% loss; fruit russeting and reduced market value",
    },
    "Cherry___healthy": {
        "crop": "Cherry", "severity_modifier": 0.0,
        "organic": [], "chemical": [],
        "preventive": [
            "Schedule pre-bloom copper spray",
            "Monitor for powdery mildew during dry windy conditions",
            "Maintain tree vigor with adequate irrigation",
        ],
        "yield_impact": "No yield impact — tree is healthy",
    },
    # ── SINGLETONS ───────────────────────────────────────────────────────────
    "Raspberry___healthy": {
        "crop": "Raspberry", "severity_modifier": 0.0,
        "organic": [], "chemical": [],
        "preventive": [
            "Prune out old floricanes after harvest",
            "Monitor for Botrytis during fruiting",
            "Maintain row spacing for air circulation",
        ],
        "yield_impact": "No yield impact — canes are healthy",
    },
    "Soybean___healthy": {
        "crop": "Soybean", "severity_modifier": 0.0,
        "organic": [], "chemical": [],
        "preventive": [
            "Inoculate seed with Bradyrhizobium before sowing",
            "Monitor for soybean rust after R1 stage",
            "Maintain weed-free conditions in first 30 days",
        ],
        "yield_impact": "No yield impact — crop is healthy",
    },
    "Squash___Powdery_mildew": {
        "crop": "Squash", "severity_modifier": 1.0,
        "organic": [
            {"method": "Baking soda (1%)",              "application": "Foliar spray", "frequency": "Every 3 days"},
            {"method": "Potassium bicarbonate (5 g/L)", "application": "Foliar spray", "frequency": "Every 5 days"},
            {"method": "Neem oil",                       "application": "Foliar",       "frequency": "Every 7 days"},
        ],
        "chemical": [
            {"product": "Sulfur 80% WP",      "dosage": "2.5 g/L", "timing": "Preventive",    "safety_note": "PHI 0 days; avoid >35°C"},
            {"product": "Azoxystrobin",       "dosage": "1 ml/L",  "timing": "At first sign", "safety_note": "PHI 0 days"},
            {"product": "Tetraconazole",      "dosage": "0.5 ml/L","timing": "Curative",       "safety_note": "PHI 3 days"},
        ],
        "preventive": [
            "Plant resistant varieties",
            "Reduce plant density for better airflow",
            "Avoid late-evening watering",
        ],
        "yield_impact": "20–40% loss; reduces fruit set",
    },
    "Blueberry___healthy": {
        "crop": "Blueberry", "severity_modifier": 0.0,
        "organic": [], "chemical": [],
        "preventive": [
            "Maintain soil pH 4.5–5.5 with sulfur amendments",
            "Mulch with pine bark to retain moisture",
            "Monitor for mummy berry in spring",
        ],
        "yield_impact": "No yield impact — bushes are healthy",
    },
    "Orange___Haunglongbing": {
        "crop": "Orange", "severity_modifier": 2.0,
        "organic": [
            {"method": "Yellow sticky traps",           "application": "To monitor psyllid vector", "frequency": "Check weekly"},
            {"method": "Reflective mulch",              "application": "Under trees",               "frequency": "Season-long"},
            {"method": "Mineral oil spray",             "application": "Foliar; smothers psyllids", "frequency": "Every 14 days"},
        ],
        "chemical": [
            {"product": "Imidacloprid",      "dosage": "0.5 ml/L", "timing": "Systemic; new flush control", "safety_note": "PHI 14 days; bee risk"},
            {"product": "Thiamethoxam",      "dosage": "0.3 g/L",  "timing": "Pre-flush spray",             "safety_note": "PHI 7 days"},
            {"product": "Dimethoate",        "dosage": "2 ml/L",   "timing": "At psyllid nymph stage",      "safety_note": "PHI 7 days; broad spectrum"},
        ],
        "preventive": [
            "REMOVE AND DESTROY infected trees — no cure exists for HLB",
            "Use disease-free certified budwood for new plantings",
            "Control Asian citrus psyllid aggressively in entire region",
            "Do not move plant material from HLB-affected areas",
            "Quarantine affected trees to prevent spread",
        ],
        "yield_impact": "TOTAL LOSS — infected trees must be removed. Disease is incurable.",
    },
}

# Fill remaining entries not explicitly listed using crop-appropriate fallbacks
_KNOWN = set(DISEASE_DB.keys())
_TEMPLATE_ORGANIC = [
    {"method": "Neem oil spray",          "application": "5 ml/L water, foliar spray",      "frequency": "Every 7 days"},
    {"method": "Trichoderma bio-fungicide","application": "2 g/L soil drench + foliar",      "frequency": "Every 14–21 days"},
    {"method": "Copper oxychloride",      "application": "2.5 g/L foliar",                   "frequency": "Every 10 days"},
]
_TEMPLATE_CHEMICAL = [
    {"product": "Mancozeb 75% WP",  "dosage": "2.5 g/L", "timing": "At first symptom",         "safety_note": "PHI 7 days; wear gloves"},
    {"product": "Chlorothalonil",    "dosage": "2 g/L",   "timing": "Preventive fortnightly",   "safety_note": "PHI 7 days; wear mask"},
]
_TEMPLATE_PREVENTIVE = [
    "Use certified disease-free planting material",
    "Maintain field hygiene — remove and destroy infected debris",
    "Practice 2-year crop rotation",
    "Balanced NPK fertilization; avoid excess nitrogen",
]

ALL_38_CLASSES = [
    "Tomato___Late_blight","Tomato___Early_blight","Tomato___Bacterial_spot",
    "Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot","Tomato___Spider_mites",
    "Tomato___Target_Spot","Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus","Tomato___healthy",
    "Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy",
    "Corn___Cercospora_leaf_spot","Corn___Common_rust","Corn___Northern_Leaf_Blight","Corn___healthy",
    "Grape___Black_rot","Grape___Esca","Grape___Leaf_blight","Grape___healthy",
    "Potato___Early_blight","Potato___Late_blight","Potato___healthy",
    "Pepper___Bacterial_spot","Pepper___healthy",
    "Strawberry___Leaf_scorch","Strawberry___healthy",
    "Peach___Bacterial_spot","Peach___healthy",
    "Cherry___Powdery_mildew","Cherry___healthy",
    "Raspberry___healthy","Soybean___healthy","Squash___Powdery_mildew",
    "Blueberry___healthy","Orange___Haunglongbing",
]

for _cls in ALL_38_CLASSES:
    if _cls not in _KNOWN:
        _crop = _cls.split("___")[0].replace("_", " ")
        DISEASE_DB[_cls] = {
            "crop": _crop, "severity_modifier": 1.0,
            "organic":   _TEMPLATE_ORGANIC,
            "chemical":  _TEMPLATE_CHEMICAL,
            "preventive": _TEMPLATE_PREVENTIVE,
            "yield_impact": "Variable yield impact depending on severity and timeliness of management",
        }

# ─────────────────────────────────────────────────────────────────────────────
# Contextual Data
# ─────────────────────────────────────────────────────────────────────────────
REGIONS = [
    "North India (Punjab/Haryana)","North India (Uttar Pradesh)","North India (Himachal Pradesh)",
    "South India (Karnataka)","South India (Tamil Nadu)","South India (Andhra Pradesh)",
    "East India (West Bengal)","East India (Odisha)","East India (Bihar)",
    "West India (Maharashtra/Vidarbha)","West India (Gujarat)","West India (Rajasthan)",
    "Central India (Madhya Pradesh)","Central India (Chhattisgarh)",
    "North-East India (Assam/Meghalaya)",
]

SEASONS = ["Kharif (Monsoon, Jun–Oct)","Rabi (Winter, Nov–Mar)","Zaid (Summer, Apr–Jun)"]

SEVERITY_LEVELS = ["Mild (0–30% infection)","Moderate (30–60% infection)","Severe (60–100% infection)"]

URGENCY_MAP = {
    "Mild (0–30% infection)":   "Within 7 days",
    "Moderate (30–60% infection)": "Within 3 days",
    "Severe (60–100% infection)": "Immediate action required",
}

FARMER_PROFILES = [
    {"type": "Small-scale subsistence farmer",  "land": "< 2 ha", "budget": "low",    "note": "Prefer low-cost organic solutions and locally available inputs."},
    {"type": "Medium-scale commercial farmer",  "land": "2–10 ha","budget": "moderate","note": "Balance between cost and efficacy; open to chemical treatments."},
    {"type": "Large-scale agribusiness operator","land": "> 10 ha","budget": "high",   "note": "Mechanized application possible; optimise for maximum yield protection."},
]

REGIONAL_NOTES = {
    "North India (Punjab/Haryana)":         "Hot dry summers; wheat-rice rotation; high groundwater use. Prioritize drip irrigation and reduce disease risk with resistant varieties.",
    "North India (Uttar Pradesh)":          "Semi-arid plains; high sugarcane and wheat acreage. Dust formulations preferred in arid pockets.",
    "North India (Himachal Pradesh)":       "Cool temperate hills; high humidity promotes fungal diseases on apples and stone fruits. Dense canopy management is critical.",
    "South India (Karnataka)":              "Bimodal rainfall; high humidity year-round promotes fungal and bacterial diseases. Increase spray frequency by 30%.",
    "South India (Tamil Nadu)":             "Tropical; high temperatures and monsoon irrigation — bacterial diseases peak during rainy season.",
    "South India (Andhra Pradesh)":         "Coastal humid climate; chilli and rice dominant; viral diseases (TYLCV) common via whitefly vectors.",
    "East India (West Bengal)":             "High rainfall; blight-prone July–September. Prophylactic copper sprays critical before monsoon onset.",
    "East India (Odisha)":                  "Tribal farming regions; emphasis on integrated pest management and locally sourced botanicals.",
    "East India (Bihar)":                   "Flood-prone Gangetic plains; post-flood disease build-up. Drainage improvement critical.",
    "West India (Maharashtra/Vidarbha)":    "Cotton and soybean dominant; dry Vidarbha experiences disease stress during patchy monsoon years.",
    "West India (Gujarat)":                 "Semi-arid; groundnut and cotton dominant; drip irrigation recommended to reduce foliar disease pressure.",
    "West India (Rajasthan)":               "Arid zone; reduced fungal pressure but severe viral and spider-mite problems under drought stress.",
    "Central India (Madhya Pradesh)":       "Black cotton soil moisture retention; soybean and wheat dominant; post-monsoon disease build-up common.",
    "Central India (Chhattisgarh)":         "Tribal and small-scale farming; rice dominant; blast and brown spot management critical.",
    "North-East India (Assam/Meghalaya)":   "High rainfall (> 2000 mm/yr); extreme fungal disease pressure; copper and systemic fungicide programs essential.",
}

SEASONAL_NOTES = {
    "Kharif (Monsoon, Jun–Oct)":  "Peak risk period for fungal and bacterial diseases. Begin spray program 2 weeks before expected monsoon onset. Avoid copper sprays during active rain — reschedule within 24 hrs post-rain.",
    "Rabi (Winter, Nov–Mar)":     "Powdery mildew peak in cool dry nights with heavy dews. Avoid overhead irrigation. Rust monitoring essential for cereals after December.",
    "Zaid (Summer, Apr–Jun)":     "Heat stress weakens plant immunity; viral diseases (whitefly-transmitted) surge. Focus on vector control; reduce fungicide pressure.",
}

# ─────────────────────────────────────────────────────────────────────────────
# 12 Instruction Templates for Diversity
# ─────────────────────────────────────────────────────────────────────────────
INSTRUCTION_TEMPLATES = [
    # 0 — Standard advisory
    lambda d,c,r,s,sev: (
        f"A farmer in {r} is growing {c} during the {s} season. "
        f"The crop has been diagnosed with {d} at a {sev} level. "
        f"Provide a complete integrated disease management plan."
    ),
    # 1 — Urgent alert
    lambda d,c,r,s,sev: (
        f"URGENT: A {sev.split('(')[0].strip().lower()} outbreak of {d} has been detected in a {c} field in {r} during {s}. "
        f"What immediate steps should the farmer take to contain the spread and protect yield?"
    ),
    # 2 — Organic farm preference
    lambda d,c,r,s,sev: (
        f"An organic {c} farmer in {r} is dealing with {d} ({sev}) during {s}. "
        f"They want to avoid synthetic chemicals. Recommend an organic-only management strategy."
    ),
    # 3 — First-time farmer
    lambda d,c,r,s,sev: (
        f"A first-time {c} farmer in {r} has noticed symptoms matching {d}. The infection appears to be {sev}. "
        f"It is currently {s}. Explain in simple terms what this disease is, how serious it is, and step-by-step what to do."
    ),
    # 4 — Chemical efficacy question
    lambda d,c,r,s,sev: (
        f"Which chemical fungicides or bactericides are most effective against {d} on {c} at {sev} infection during {s}? "
        f"The farm is located in {r}. Include dosage, timing and safety information."
    ),
    # 5 — Season-focused
    lambda d,c,r,s,sev: (
        f"How does the {s} season affect the management of {d} on {c} crops in {r}? "
        f"The current infection level is {sev}. Provide a season-adjusted treatment protocol."
    ),
    # 6 — Preventive only
    lambda d,c,r,s,sev: (
        f"A {c} farmer in {r} has successfully managed {d} this {s}. "
        f"What preventive measures should they implement next season to avoid recurrence? (Previous severity was {sev}.)"
    ),
    # 7 — Comparative question
    lambda d,c,r,s,sev: (
        f"Compare organic vs chemical treatment options for {d} affecting {c} crops at {sev} severity in {r} during {s}. "
        f"Include cost-effectiveness and crop safety considerations."
    ),
    # 8 — Diagnosis + management
    lambda d,c,r,s,sev: (
        f"The AI vision system has identified {d} on a {c} crop in {r}. Severity is {sev}. The current season is {s}. "
        f"Confirm the diagnosis, assess the risk, and provide a detailed management plan."
    ),
    # 9 — Small farmer budget
    lambda d,c,r,s,sev: (
        f"A small-scale {c} farmer in {r} with a limited budget has {sev} {d} during {s}. "
        f"What are the most cost-effective treatments using locally available inputs?"
    ),
    # 10 — Extension officer perspective
    lambda d,c,r,s,sev: (
        f"As an agricultural extension officer advising farmers in {r} during {s}, "
        f"what guidance would you provide for managing {d} on {c} at {sev} infection level?"
    ),
    # 11 — Market/economic angle
    lambda d,c,r,s,sev: (
        f"A commercial {c} grower in {r} is worried that {d} ({sev}) detected during {s} will impact harvest quality. "
        f"What is the expected yield impact and what is the most economically justified treatment strategy?"
    ),
]

INPUT_TEMPLATES = [
    # short
    lambda d,c,r,s,sev: f"Disease: {d}\nCrop: {c}\nRegion: {r}\nSeason: {s}\nSeverity: {sev}",
    # with context
    lambda d,c,r,s,sev: f"Diagnosed disease: {d}\nAffected crop: {c}\nFarm region: {r}\nGrowing season: {s}\nInfection severity: {sev}",
    # minimal
    lambda d,c,r,s,sev: f"{c} | {d} | {sev} | {r} | {s}",
]

# ─────────────────────────────────────────────────────────────────────────────
# Pair Generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_pair(disease: str, region: str, season: str, severity: str, farmer: dict, rng: random.Random) -> dict:
    db   = DISEASE_DB[disease]
    crop = db["crop"]
    d_pretty = disease.replace("___", " — ").replace("_", " ")

    instruction_fn = rng.choice(INSTRUCTION_TEMPLATES)
    input_fn       = rng.choice(INPUT_TEMPLATES)

    instruction = instruction_fn(d_pretty, crop, region, season, severity)
    input_ctx   = input_fn(d_pretty, crop, region, season, severity)

    # Selectively strip chemical recommendations for organic-template responses
    organic_only = "organic-only" in instruction.lower()
    chemical_list = [] if organic_only else db["chemical"]

    output = {
        "disease_confirmed":        d_pretty,
        "crop":                     crop,
        "severity_assessment":      severity,
        "urgency":                  URGENCY_MAP[severity],
        "farmer_profile":           f"{farmer['type']} ({farmer['land']})",
        "organic_treatments":       db["organic"],
        "chemical_treatments":      chemical_list,
        "preventive_measures":      db["preventive"],
        "yield_impact_estimate":    db["yield_impact"],
        "regional_advisory":        REGIONAL_NOTES[region],
        "seasonal_advisory":        SEASONAL_NOTES[season],
        "budget_consideration":     farmer["note"],
    }

    return {
        "instruction": instruction,
        "input":       input_ctx,
        "output":      json.dumps(output, ensure_ascii=False),
    }


def generate_dataset(n: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    pairs = []
    for i in range(n):
        pairs.append(generate_pair(
            disease  = rng.choice(ALL_38_CLASSES),
            region   = rng.choice(REGIONS),
            season   = rng.choice(SEASONS),
            severity = rng.choice(SEVERITY_LEVELS),
            farmer   = rng.choice(FARMER_PROFILES),
            rng      = rng,
        ))
    return pairs


def print_stats(pairs: list):
    crops, severities, templates = Counter(), Counter(), Counter()
    lengths = []
    for p in pairs:
        out = json.loads(p["output"])
        crops[out["crop"]] += 1
        severities[out["severity_assessment"].split("(")[0].strip()] += 1
        lengths.append(len(p["instruction"]) + len(p["input"]) + len(p["output"]))
        # Identify template type from instruction content
        if "URGENT" in p["instruction"]:           templates["urgent"] += 1
        elif "organic-only" in p["instruction"].lower(): templates["organic_only"] += 1
        elif "extension officer" in p["instruction"]: templates["extension"] += 1
        elif "commercial" in p["instruction"]:      templates["commercial"] += 1
        else:                                       templates["standard"] += 1

    print("\n── Dataset Statistics ──────────────────────────────────")
    print(f"  Total pairs    : {len(pairs):,}")
    print(f"  Diseases       : {len(set(json.loads(p['output'])['disease_confirmed'] for p in pairs))}")
    print(f"  Regions        : {len(REGIONS)}")
    print(f"  Templates      : {len(INSTRUCTION_TEMPLATES)} instruction styles")
    print(f"\n  Top crops by pair count:")
    for crop, cnt in crops.most_common(8):
        print(f"    {crop:<25} {cnt:>6,}")
    print(f"\n  Severity distribution:")
    for sev, cnt in severities.most_common():
        print(f"    {sev:<12} {cnt:>6,}")
    print(f"\n  Instruction styles (approx):")
    for tpl, cnt in templates.most_common():
        print(f"    {tpl:<20} {cnt:>6,}")
    print(f"\n  Avg char length: {sum(lengths)/len(lengths):.0f}")
    print(f"  Min / Max      : {min(lengths)} / {max(lengths)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int, default=50_000, help="Number of instruction pairs")
    parser.add_argument("--output", default="data/processed/llm_instructions.jsonl")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--stats",  action="store_true", help="Print dataset statistics after generation")
    args = parser.parse_args()

    print(f"[1/2] Generating {args.n:,} instruction pairs (seed={args.seed})...")
    pairs = generate_dataset(args.n, seed=args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[2/2] Saving to {out_path} ...")
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print_stats(pairs)
    print(f"\n✅ Dataset saved: {out_path}  ({out_path.stat().st_size / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
