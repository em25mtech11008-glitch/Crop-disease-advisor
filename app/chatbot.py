"""
CropMind AI Chatbot — 100% powered by fine-tuned QLoRA model.
Every response streams directly from Qwen2.5-3B + LoRA adapter.
Run: streamlit run app/chatbot.py --server.port 8502
"""
import sys
import time
import torch
from pathlib import Path
from threading import Thread
import streamlit as st
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropMind AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ─────────────────────────────────────────────────────────────────────────────
# Theme state — must come before CSS injection
# ─────────────────────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
DARK = st.session_state.theme == "dark"
# ─────────────────────────────────────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────────────────────────────────────
if DARK:
    BG          = "#050a06"
    BG2         = "#081210"
    BG3         = "#0b1a13"
    SURFACE     = "#0d1f16"
    BORDER      = "#164430"
    BORDER_GLOW = "#34d39955"
    TEXT        = "#e2f5ea"
    TEXT_MUT    = "#6b9b7e"
    TEXT_DIM    = "#2f5e42"
    ACCENT      = "#34d399"
    ACCENT2     = "#38bdf8"
    ACCENT3     = "#a78bfa"
    USER_BG     = "linear-gradient(135deg,#0c2516 0%,#0e2e1e 100%)"
    USER_BD     = "#34d39925"
    BOT_BG      = "linear-gradient(135deg,#081218 0%,#091820 100%)"
    BOT_BD      = "#38bdf815"
    SIDEBAR_BG  = "#050a06"
    INPUT_BG    = "#081210"
    INPUT_BD    = "#164430"
    STAT_BG     = "#0d1f16"
    CODE_BG     = "#0b1a13"
else:
    BG          = "#f4faf6"
    BG2         = "#ffffff"
    BG3         = "#eaf5ef"
    SURFACE     = "#ffffff"
    BORDER      = "#c4e0d0"
    BORDER_GLOW = "#16a34a44"
    TEXT        = "#15302a"
    TEXT_MUT    = "#3d7a5c"
    TEXT_DIM    = "#88b8a0"
    ACCENT      = "#059669"
    ACCENT2     = "#0284c7"
    ACCENT3     = "#7c3aed"
    USER_BG     = "linear-gradient(135deg,#e0f9ec 0%,#d5f5e3 100%)"
    USER_BD     = "#34d39940"
    BOT_BG      = "linear-gradient(135deg,#ffffff 0%,#f8fdfa 100%)"
    BOT_BD      = "#bae6fd"
    SIDEBAR_BG  = "#f4faf6"
    INPUT_BG    = "#ffffff"
    INPUT_BD    = "#c4e0d0"
    STAT_BG     = "#eaf5ef"
    CODE_BG     = "#e0f9ec"
# ─────────────────────────────────────────────────────────────────────────────
# CSS injection
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
*, html, body, [class*="css"] {{
    font-family: 'Outfit', sans-serif !important;
    box-sizing: border-box;
}}
/* ─── App shell ─── */
.stApp {{
    background: {BG} !important;
    {"background-image: radial-gradient(ellipse 80% 50% at 50% -5%, #0a2e1844 0%, transparent 50%), radial-gradient(ellipse 40% 35% at 90% 90%, #0e1a3033 0%, transparent 45%) !important;" if DARK else ""}
    min-height: 100vh;
}}
#MainMenu, footer, header {{ visibility: hidden !important; }}
.block-container {{ padding: 0 !important; max-width: 100% !important; }}
/* hide streamlit deploy button */
.stDeployButton {{ display: none !important; }}
/* kill all white background leaks */
.stApp > div, .stApp > div > div, .main .block-container,
[data-testid="stBottomBlockContainer"],
[data-testid="stBottom"] > div,
section.main > div {{
    background: {BG} !important;
}}
/* ─── Sidebar ─── */
[data-testid="stSidebar"] {{
    background: {SIDEBAR_BG} !important;
    border-right: 1px solid {BORDER}44 !important;
}}
[data-testid="stSidebar"] > div {{
    padding: 1.2rem 1rem !important;
}}
/* ─── Buttons ─── */
div[data-testid="stButton"] > button {{
    background: {"linear-gradient(135deg,#0d1f16,#0e2a1c)" if DARK else "linear-gradient(135deg,#f4faf6,#eaf5ef)"} !important;
    border: 1px solid {BORDER}55 !important;
    color: {TEXT_MUT} !important;
    border-radius: 12px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    text-align: left !important;
    padding: 0.55rem 1rem !important;
    width: 100% !important;
    margin-bottom: 0.35rem !important;
    transition: all 0.22s cubic-bezier(.4,0,.2,1) !important;
    letter-spacing: 0.01em !important;
}}
div[data-testid="stButton"] > button:hover {{
    background: {"linear-gradient(135deg,#14532d,#166534)" if DARK else "linear-gradient(135deg,#d5f5e3,#c8f0da)"} !important;
    border-color: {ACCENT}66 !important;
    color: {ACCENT} !important;
    transform: translateX(4px) !important;
    box-shadow: {"0 4px 20px #34d39912, 0 0 0 1px #34d39908" if DARK else "0 4px 16px #22c55e15"} !important;
}}
/* ─── Chat messages ─── */
[data-testid="stChatMessage"] {{
    background: transparent !important;
    border: none !important;
    padding: 0.4rem 0 !important;
}}
[data-testid="stChatMessage"] > div {{
    border-radius: 18px !important;
    padding: 1.1rem 1.4rem !important;
    transition: all 0.25s cubic-bezier(.4,0,.2,1);
    {"backdrop-filter: blur(12px);" if DARK else ""}
}}
/* user */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div {{
    background: {USER_BG} !important;
    border: 1px solid {USER_BD} !important;
    border-radius: 22px 22px 4px 22px !important;
    max-width: 78% !important;
    margin-left: auto !important;
    box-shadow: {"0 8px 32px #34d39908, inset 0 1px 0 #34d39910" if DARK else "0 4px 20px #22c55e12"} !important;
}}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div:hover {{
    box-shadow: {"0 8px 40px #34d39915" if DARK else "0 6px 24px #22c55e18"} !important;
}}
/* assistant */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) > div {{
    background: {BOT_BG} !important;
    border: 1px solid {BOT_BD} !important;
    border-radius: 22px 22px 22px 4px !important;
    box-shadow: {"0 8px 32px #38bdf806, inset 0 1px 0 #38bdf808" if DARK else "0 4px 20px #0891b210"} !important;
}}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) > div:hover {{
    box-shadow: {"0 8px 40px #38bdf812" if DARK else "0 6px 24px #0891b215"} !important;
}}
/* text in messages */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {{
    color: {TEXT} !important;
    font-size: 0.93rem !important;
    line-height: 1.75 !important;
}}
[data-testid="stChatMessage"] strong {{
    color: {ACCENT} !important;
    font-weight: 600 !important;
}}
[data-testid="stChatMessage"] code {{
    background: {CODE_BG} !important;
    color: {ACCENT} !important;
    border: 1px solid {BORDER}55 !important;
    border-radius: 5px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    padding: 0.15rem 0.5rem !important;
}}
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3 {{
    color: {ACCENT} !important;
    font-weight: 700 !important;
    margin-top: 0.8rem !important;
    margin-bottom: 0.3rem !important;
}}
[data-testid="stChatMessage"] ul {{
    padding-left: 1.2rem !important;
}}
[data-testid="stChatMessage"] li {{
    margin-bottom: 0.25rem !important;
}}
[data-testid="stChatMessage"] hr {{
    border-color: {BORDER}33 !important;
    margin: 1rem 0 !important;
}}
[data-testid="stChatMessage"] table {{
    border-collapse: separate !important;
    border-spacing: 0 !important;
    border: 1px solid {BORDER}33 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    width: 100% !important;
    margin: 0.6rem 0 !important;
}}
[data-testid="stChatMessage"] th {{
    background: {SURFACE} !important;
    color: {TEXT_DIM} !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    padding: 0.5rem 0.9rem !important;
    border-bottom: 1px solid {BORDER}33 !important;
}}
[data-testid="stChatMessage"] td {{
    padding: 0.55rem 0.9rem !important;
    border-bottom: 1px solid {BORDER}18 !important;
    font-size: 0.88rem !important;
    color: {TEXT} !important;
}}
[data-testid="stChatMessage"] tr:last-child td {{
    border-bottom: none !important;
}}
[data-testid="stChatMessage"] blockquote {{
    border-left: 3px solid {ACCENT}66 !important;
    background: {ACCENT}08 !important;
    border-radius: 0 8px 8px 0 !important;
    padding: 0.7rem 1rem !important;
    margin: 0.8rem 0 !important;
    font-size: 0.88rem !important;
}}
[data-testid="stChatMessage"] blockquote p {{
    margin: 0 !important;
}}
/* ─── Chat input ─── */
.stChatInputContainer {{
    background: {"linear-gradient(180deg,transparent," + BG + "cc," + BG + ")" if DARK else BG} !important;
    border-top: 1px solid {BORDER}22 !important;
    padding: 1rem 2rem 1.2rem !important;
    position: sticky !important;
    bottom: 0 !important;
    {"backdrop-filter: blur(16px);" if DARK else ""}
}}
[data-testid="stChatInput"] > div {{
    background: {INPUT_BG} !important;
    border: 1.5px solid {INPUT_BD} !important;
    border-radius: 24px !important;
    transition: all 0.25s cubic-bezier(.4,0,.2,1) !important;
    box-shadow: {"0 4px 24px #00000020, inset 0 1px 0 " + BORDER + "22" if DARK else "0 2px 12px #00000008"} !important;
}}
[data-testid="stChatInput"] > div > div {{
    border: none !important;
    background: transparent !important;
}}
[data-testid="stChatInput"] {{
    border: none !important;
    background: transparent !important;
}}
[data-testid="stChatInput"] * {{
    border-color: {INPUT_BD} !important;
}}
[data-testid="stChatInput"] > div:focus-within {{
    border-color: {ACCENT}88 !important;
    box-shadow: 0 0 0 3px {ACCENT}12, 0 4px 32px {ACCENT}08 !important;
}}
[data-testid="stChatInput"] textarea {{
    color: {TEXT} !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.93rem !important;
    background: transparent !important;
    line-height: 1.6 !important;
}}
[data-testid="stChatInput"] textarea::placeholder {{
    color: {TEXT_DIM} !important;
}}
/* ─── Metrics ─── */
[data-testid="stMetric"] {{
    background: {STAT_BG} !important;
    border: 1px solid {BORDER}44 !important;
    border-radius: 10px !important;
    padding: 0.6rem 0.8rem !important;
}}
[data-testid="stMetricLabel"] {{ color: {TEXT_MUT} !important; font-size: 0.72rem !important; }}
[data-testid="stMetricValue"] {{ color: {ACCENT} !important; font-size: 1.2rem !important; font-weight: 700 !important; font-family: 'JetBrains Mono', monospace !important; }}
/* ─── Alerts ─── */
.stAlert {{
    border-radius: 12px !important;
    font-size: 0.88rem !important;
}}
/* ─── Scrollbar ─── */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}88; border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: {ACCENT}66; }}
/* ─── Animations ─── */
@keyframes fade-in {{ from {{ opacity:0; transform:translateY(10px); }} to {{ opacity:1; transform:translateY(0); }} }}
@keyframes pulse-glow {{
    0%,100% {{ box-shadow: 0 0 6px {ACCENT}33; }}
    50% {{ box-shadow: 0 0 18px {ACCENT}55; }}
}}
@keyframes live-pulse {{
    0%,100% {{ opacity:1; transform:scale(1); }}
    50% {{ opacity:0.3; transform:scale(0.7); }}
}}
@keyframes gradient-shift {{
    0% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}
@keyframes shimmer {{
    0% {{ background-position: -200% 0; }}
    100% {{ background-position: 200% 0; }}
}}
[data-testid="stChatMessage"] {{
    animation: fade-in 0.35s cubic-bezier(.4,0,.2,1) !important;
}}
/* ─── Custom components ─── */
.cm-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.1rem 2rem 0.9rem;
    border-bottom: 1px solid {BORDER}25;
    background: {"linear-gradient(180deg," + BG + "ee," + BG + "cc)" if DARK else BG};
    position: sticky;
    top: 0;
    z-index: 100;
    {"backdrop-filter: blur(20px);" if DARK else "backdrop-filter: blur(12px);"}
}}
.cm-logo {{
    display: flex;
    align-items: center;
    gap: 0.7rem;
}}
.cm-logo-icon {{
    width: 36px; height: 36px;
    border-radius: 10px;
    background: linear-gradient(135deg, {ACCENT}30, {ACCENT2}20);
    border: 1px solid {ACCENT}44;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    {"box-shadow: 0 0 16px " + ACCENT + "22;" if DARK else ""}
}}
.cm-logo-text {{
    font-size: 1.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, {ACCENT}, {ACCENT2}, {ACCENT});
    background-size: 200% 100%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    animation: gradient-shift 6s ease infinite;
}}
.cm-logo-sub {{
    font-size: 0.62rem;
    color: {TEXT_DIM};
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.12em;
    margin-top: 0.1rem;
}}
.live-pill {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: {ACCENT}10;
    border: 1px solid {ACCENT}33;
    border-radius: 20px;
    padding: 0.3rem 0.8rem;
    font-size: 0.7rem;
    color: {ACCENT};
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
    {"animation: pulse-glow 3s infinite;" if DARK else ""}
}}
.live-dot {{
    width: 6px; height: 6px;
    border-radius: 50%;
    background: {ACCENT};
    {"box-shadow: 0 0 6px " + ACCENT + ";" if DARK else ""}
    animation: live-pulse 2s infinite;
}}
.cm-badges {{
    display: flex;
    gap: 0.5rem;
    align-items: center;
}}
.model-pill {{
    background: {ACCENT3}18;
    border: 1px solid {ACCENT3}33;
    border-radius: 20px;
    padding: 0.3rem 0.8rem;
    font-size: 0.68rem;
    color: {ACCENT3};
    font-family: 'JetBrains Mono', monospace;
}}
.sb-label {{
    font-size: 0.65rem;
    font-weight: 700;
    color: {TEXT_DIM};
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    margin: 1.2rem 0 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}}
.sb-label::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: {BORDER}44;
}}
.model-box {{
    background: {"linear-gradient(135deg,#0a1a0e 0%,#071210 100%)" if DARK else "linear-gradient(135deg,#f8fffe 0%,#f0faf4 100%)"};
    border: 1px solid {BORDER}44;
    border-radius: 14px;
    padding: 1.1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s ease;
}}
.model-box:hover {{
    border-color: {ACCENT}44;
}}
{"" if not DARK else """.model-box::before {{
    content: '';
    position: absolute;
    top: 0; left: -100%; width: 200%; height: 100%;
    background: linear-gradient(90deg, transparent, #34d39908, transparent);
    animation: shimmer 4s ease infinite;
}}"""}
.model-name {{
    font-size: 0.85rem;
    font-weight: 700;
    color: {ACCENT};
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.2rem;
}}
.model-adapter {{
    font-size: 0.72rem;
    color: {TEXT_MUT};
    margin-bottom: 0.7rem;
}}
.spec-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.28rem 0;
    border-bottom: 1px solid {BORDER}33;
    font-size: 0.73rem;
}}
.spec-row:last-child {{ border-bottom: none; }}
.spec-k {{ color: {TEXT_DIM}; }}
.spec-v {{
    color: {ACCENT};
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
}}
.topic-btn-label {{
    font-size: 0.78rem;
    color: {TEXT_MUT};
}}
/* ─── Main chat scroll area ─── */
.main-chat {{
    padding: 1.2rem 2rem 1rem;
    max-width: 900px;
    margin: 0 auto;
}}
</style>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────────────────
# System prompts — two different ones for each model mode
# ─────────────────────────────────────────────────────────────────────────────

# Used with base model (adapters OFF) — handles general conversation
CHAT_SYSTEM = """You are CropMind, a warm and expert agricultural advisor for Indian farmers.
Your job is to gather enough information and wait for the user to request a treatment plan before giving one.

RULES:
- If the farmer hasn't told you their CROP, ask for it.
- If the farmer hasn't described any SYMPTOMS or DISEASE, ask for that.
- If the farmer hasn't told you their REGION, ask for it.
- If the farmer hasn't given the SEVERITY (e.g. Mild, Moderate, Severe, or percentage), ask for it.
- Ask only ONE question at a time — never ask multiple things at once.
- DO NOT generate a treatment plan until the user specifically asks for it AND you have all 4 pieces of information.
- If they ask for a plan but information is missing, politely tell them what is missing and ask for it.
- Keep replies SHORT (2-4 lines max).
- Be warm, patient, and encouraging.

EXAMPLES:
User: "I'm from West India"
Assistant: "Welcome! Which crop are you growing this season? 🌱"

User: "I grow tomatoes"
Assistant: "Great! What symptoms are you seeing on your tomatoes? (e.g. spots, wilting, color change)"

User: "potatoes, leaves turning black and rotting, moderate severity in punjab. Please generate a report."
Assistant: "Got it! Generating your treatment plan report now..."""

# Used with fine-tuned model (adapters ON) — generates structured treatment plans
PLAN_SYSTEM = """You are CropMind, an expert agricultural advisor. The farmer has already told you their crop and symptoms.
Now generate a complete, structured treatment plan.

Always use this exact format:
## 🔍 Diagnosis
[Disease name and brief explanation]

## 🌱 Organic Treatments
- [Method]: [How to apply] — [Frequency]

## 🧪 Chemical Treatments
- [Product name]: [Dosage] — [Safety note]

## 🛡️ Prevention
- [Tip]

## ⏰ Action Urgency
[Timeframe and priority]

Be specific with product names, dosages, and timing. Keep language simple."""

# default starting history for the chat mode
SYSTEM_PROMPT = CHAT_SYSTEM  # used for session init
# ─────────────────────────────────────────────────────────────────────────────
# Model loading — cached once for the whole session
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    base_model_id = "Qwen/Qwen2.5-3B-Instruct"
    adapter_path  = "models/llm/qwen2.5_3b_qlora_adapter"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    device = str(next(model.parameters()).device)
    return tokenizer, model, device


# ─────────────────────────────────────────────────────────────────────────────
# Context detector — decides if we have enough to generate a treatment plan
# ─────────────────────────────────────────────────────────────────────────────
import re

KNOWN_CROPS = [
    "tomato", "tomatoes", "potato", "potatoes", "wheat", "rice", "corn", "maize", "grape", "grapes", "apple", "apples",
    "pepper", "peppers", "onion", "onions", "mango", "mangoes", "cotton", "soybean", "sugarcane", "banana", "bananas",
    "chilli", "brinjal", "eggplant", "cauliflower", "cabbage", "groundnut",
    "sunflower", "mustard", "garlic", "ginger", "turmeric", "lemon", "orange",
    "strawberry", "cherry", "peach", "squash", "cucumber", "watermelon",
]

SYMPTOM_WORDS = [
    "spot", "spots", "blight", "rust", "rot", "rotting", "wilt", "wilting", "yellow", "yellowing", "brown", "black",
    "mold", "mildew", "fungus", "lesion", "lesions", "curl", "curling", "die", "dead", "dying",
    "blast", "discolor", "discolored", "disease", "infect", "infected", "damage", "damaged",
    "patch", "patches", "stripe", "stripes", "burn", "burnt", "dry", "wet", "soft", "hard", "symptom", "symptoms",
]

def context_has_crop(history: list) -> bool:
    """Check if any message in the conversation mentions a known crop using word boundaries."""
    text = " ".join(m["content"].lower() for m in history if m["role"] == "user")
    pattern = re.compile(r'\b(' + '|'.join(KNOWN_CROPS) + r')\b')
    return bool(pattern.search(text))

def context_has_symptoms(history: list) -> bool:
    """Check if any message mentions disease symptoms using word boundaries."""
    text = " ".join(m["content"].lower() for m in history if m["role"] == "user")
    pattern = re.compile(r'\b(' + '|'.join(SYMPTOM_WORDS) + r')\b')
    return bool(pattern.search(text))

def context_has_region(history: list) -> bool:
    """Check if any message mentions a region or Indian state."""
    text = " ".join(m["content"].lower() for m in history if m["role"] == "user")
    regions = ["india", "north", "south", "east", "west", "central", "state", "pradesh", "gujarat", "maharashtra", "punjab", "haryana", "kerala", "tamil nadu", "karnataka", "bengal", "assam", "bihar", "rajasthan", "odisha", "andhra"]
    pattern = re.compile(r'\b(' + '|'.join(regions) + r')\b')
    return bool(pattern.search(text))

def context_has_severity(history: list) -> bool:
    """Check if any message mentions disease severity or percentage."""
    text = " ".join(m["content"].lower() for m in history if m["role"] == "user")
    if "%" in text:
        return True
    pattern = re.compile(r'\b(mild|moderate|severe|high|low|percent|percentage)\b')
    return bool(pattern.search(text))

def ready_for_plan(history: list) -> bool:
    """Returns True only when we have crop, symptoms, region, and severity."""
    return context_has_crop(history) and context_has_symptoms(history) and context_has_region(history) and context_has_severity(history)


# ─────────────────────────────────────────────────────────────────────────────
# Streaming inference — uses correct mode based on context
# ─────────────────────────────────────────────────────────────────────────────
def stream_response(messages: list, tokenizer, model, use_ft: bool):
    """
    use_ft=False → disable LoRA adapters, use base Qwen for conversation.
    use_ft=True  → enable LoRA adapters, use fine-tuned model for treatment plan.
    """
    from transformers import TextIteratorStreamer

    # switch model mode
    if use_ft:
        model.enable_adapter_layers()
        max_new_tokens = 1200   # full structured treatment plan needs space
        temperature    = 0.65
    else:
        model.disable_adapter_layers()
        max_new_tokens = 800    # enough for detailed explanations in chat mode
        temperature    = 0.70

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=60.0
    )
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.92,
        repetition_penalty=1.1,
        do_sample=True,
        streamer=streamer,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    Thread(target=model.generate, kwargs=gen_kwargs, daemon=True).start()
    for chunk in streamer:
        yield chunk

    # always re-enable adapters after use
    model.enable_adapter_layers()
# ─────────────────────────────────────────────────────────────────────────────
# Vision model — EfficientNet-B4, loaded separately from the LLM
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_vision_model():
    """Load EfficientNet-B4 classifier + class names. Cached for the session."""
    import json
    from src.vision.model      import EfficientNetB4Classifier
    from src.vision.preprocess import VAL_TRANSFORM

    class_names_path  = "data/processed/class_names.json"
    vision_model_path = "models/vision/efficientnet_b4_best.pt"

    with open(class_names_path) as f:
        mapping = json.load(f)
        # Critical Fix: PyTorch ImageFolder sorts folders lexicographically.
        sorted_folder_names = sorted(list(mapping.keys()), key=str)
        class_names = [mapping[folder] for folder in sorted_folder_names]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vision = EfficientNetB4Classifier(num_classes=len(class_names)).to(device)
    ckpt = torch.load(vision_model_path, map_location=device)
    vision.load_state_dict(ckpt["state_dict"])
    vision.eval()
    return vision, class_names, VAL_TRANSFORM, device


def run_vision(image, vision, class_names, transform, device):
    """Run EfficientNet-B4 on a PIL image. Returns top-5 predictions."""
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(vision(tensor), dim=1)[0]
    top5_idx  = probs.topk(5).indices.tolist()
    top5_conf = probs.topk(5).values.tolist()
    top5 = [
        {"label": class_names[i], "conf": round(c * 100, 1)}
        for i, c in zip(top5_idx, top5_conf)
    ]
    raw      = top5[0]["label"]
    crop     = raw.split("___")[0].replace("_", " ").title() if "___" in raw else raw
    disease  = raw.split("___")[1].replace("_", " ").title() if "___" in raw else raw
    conf     = top5[0]["conf"]
    severity = "Severe (60–100%)" if conf >= 80 else "Moderate (30–60%)" if conf >= 50 else "Mild (0–30%)"
    return crop, disease, conf, severity, top5


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "messages"    not in st.session_state: st.session_state.messages    = []
if "llm_history" not in st.session_state: st.session_state.llm_history = [{"role":"system","content":CHAT_SYSTEM}]
if "token_count" not in st.session_state: st.session_state.token_count = 0
if "msg_count"   not in st.session_state: st.session_state.msg_count   = 0
if "resp_times"  not in st.session_state: st.session_state.resp_times  = []
if "plan_mode"   not in st.session_state: st.session_state.plan_mode   = False
if "photo_done"  not in st.session_state: st.session_state.photo_done  = False

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # logo
    st.markdown(f"""
    <div style="text-align:center;padding:0.8rem 0 1.4rem;">
      <div style="font-size:2.8rem;margin-bottom:0.6rem;
        {'filter:drop-shadow(0 0 16px #34d39966);' if DARK else ''}">🌿</div>
      <div style="font-size:1.2rem;font-weight:800;
        background:linear-gradient(135deg,{ACCENT},{ACCENT2},{ACCENT});
        background-size:200% 100%;
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;letter-spacing:-0.02em;">CropMind AI</div>
      <div style="font-size:0.62rem;color:{TEXT_DIM};
        font-family:'JetBrains Mono',monospace;margin-top:0.3rem;
        letter-spacing:0.12em;">
        AGRICULTURAL INTELLIGENCE
      </div>
    </div>
    """, unsafe_allow_html=True)
    # ── theme toggle ──
    st.markdown(f'<div class="sb-label">🎨 Appearance</div>', unsafe_allow_html=True)
    col_d, col_l = st.columns(2)
    with col_d:
        if st.button("🌙  Dark", key="btn_dark",
                     help="Dark mode",
                     disabled=DARK):
            st.session_state.theme = "dark"
            st.rerun()
    with col_l:
        if st.button("☀️  Light", key="btn_light",
                     help="Light mode",
                     disabled=not DARK):
            st.session_state.theme = "light"
            st.rerun()
    # ── model info ──
    st.markdown(f'<div class="sb-label">⚙️ Model</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="model-box">
      <div class="model-name">🧠 CropMind Engine</div>
      <div class="model-adapter">Qwen2.5-3B + QLoRA Fine-tuned</div>
      <div class="spec-row"><span class="spec-k">Precision</span><span class="spec-v">4-bit NF4</span></div>
      <div class="spec-row"><span class="spec-k">LoRA rank</span><span class="spec-v">r = 16</span></div>
      <div class="spec-row"><span class="spec-k">Training data</span><span class="spec-v">12K pairs</span></div>
      <div class="spec-row"><span class="spec-k">Coverage</span><span class="spec-v">38 diseases</span></div>
      <div class="spec-row"><span class="spec-k">Max output</span><span class="spec-v">1200 tok</span></div>
    </div>
    """, unsafe_allow_html=True)
    # ── session stats ──
    st.markdown(f'<div class="sb-label">📊 Session</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Messages", st.session_state.msg_count)
    c2.metric("Tokens~", st.session_state.token_count)
    if st.session_state.resp_times:
        avg_t = sum(st.session_state.resp_times) / len(st.session_state.resp_times)
        st.markdown(f"""
        <div style="background:{STAT_BG};border:1px solid {BORDER}44;border-radius:10px;
          padding:0.5rem 0.8rem;margin-top:0.3rem;font-size:0.75rem;">
          <span style="color:{TEXT_DIM};">Avg response</span>
          <span style="float:right;color:{ACCENT};font-family:'JetBrains Mono',monospace;">
            {avg_t:.1f}s
          </span>
        </div>
        """, unsafe_allow_html=True)
    # ── image upload (vision model) ──
    st.markdown(f'<div class="sb-label">📸 Upload Crop Photo</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{STAT_BG};border:1px solid {BORDER}44;
      border-radius:10px;padding:0.7rem;font-size:0.75rem;color:{TEXT_DIM};
      margin-bottom:0.5rem;">
      💡 Upload a leaf/plant image — EfficientNet-B4 will auto-detect the disease
      and pass it straight to the LLM for a treatment plan.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"],
        key="img_upload", label_visibility="collapsed",
    )

    if uploaded_file and not st.session_state.photo_done:
        from PIL import Image as PILImage
        pil_img = PILImage.open(uploaded_file)
        st.image(pil_img, use_column_width=True, caption="Uploaded image")

        with st.spinner("🧠 Analysing with EfficientNet-B4..."):
            try:
                vis, cls_names, transform, vis_device = load_vision_model()
                crop, disease, conf, severity, top5 = run_vision(
                    pil_img, vis, cls_names, transform, vis_device
                )

                # confidence bar color
                bar_color = (
                    "#ef4444" if conf >= 80 else
                    "#f97316" if conf >= 50 else "#22c55e"
                )

                st.markdown(f"""
                <div style="background:{SURFACE if DARK else '#fff'};border:1px solid {BORDER}55;
                  border-radius:12px;padding:0.9rem;margin-top:0.4rem;">
                  <div style="font-size:0.75rem;color:{TEXT_DIM};margin-bottom:0.3rem;
                    font-family:'JetBrains Mono',monospace;">DETECTION RESULT</div>
                  <div style="font-size:0.92rem;font-weight:700;color:{ACCENT};
                    margin-bottom:0.15rem;">{disease}</div>
                  <div style="font-size:0.75rem;color:{TEXT_MUT};margin-bottom:0.6rem;">Crop: {crop}</div>
                  <div style="font-size:0.7rem;color:{TEXT_DIM};margin-bottom:0.25rem;">Confidence</div>
                  <div style="background:{BORDER}33;border-radius:4px;height:6px;margin-bottom:0.3rem;">
                    <div style="width:{conf}%;height:6px;border-radius:4px;
                      background:linear-gradient(90deg,{bar_color},{bar_color}aa);"></div>
                  </div>
                  <div style="font-size:0.8rem;color:{bar_color};font-family:'JetBrains Mono',monospace;
                    font-weight:600;">{conf:.1f}% &nbsp;&nbsp;
                    <span style="color:{TEXT_DIM};font-weight:400;font-size:0.7rem;">{severity}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # inject into chat as the first user message
                inject_msg = (
                    f"📸 **Photo uploaded.** EfficientNet-B4 detected:\n\n"
                    f"- **Crop:** {crop}\n"
                    f"- **Disease:** {disease}\n"
                    f"- **Confidence:** {conf:.1f}%\n"
                    f"- **Severity:** {severity}\n\n"
                )

                st.session_state._inject = inject_msg
                st.session_state.photo_done = True  # prevent re-processing on rerun
                st.rerun()

            except Exception as e:
                st.error(f"Vision model error: `{e}`")
                st.info("Make sure `models/vision/efficientnet_b4_best.pt` and `data/processed/class_names.json` exist.")


    st.markdown(f'<div class="sb-label">💬 Quick Topics</div>', unsafe_allow_html=True)
    quick_topics = [
        ("🍅", "Tomato yellow + brown spots"),
        ("🌾", "Wheat orange-rust on leaves"),
        ("🥔", "Potato wilting suddenly"),
        ("🌽", "Corn grey leaf patches"),
        ("🍇", "Grape black spot disease"),
        ("🌿", "Rice neck rot / blast"),
        ("🧅", "Onion purple blotch"),
        ("🫑", "Pepper leaves curling"),
    ]
    for emoji, topic in quick_topics:
        if st.button(f"{emoji}  {topic}", key=f"qt_{topic[:14]}"):
            st.session_state._inject = topic
            st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️  Clear chat", key="clear"):
        st.session_state.messages    = []
        st.session_state.llm_history = [{"role":"system","content":CHAT_SYSTEM}]
        st.session_state.token_count = 0
        st.session_state.msg_count   = 0
        st.session_state.resp_times  = []
        st.session_state.plan_mode   = False
        st.session_state.photo_done  = False
        st.rerun()
# ─────────────────────────────────────────────────────────────────────────────
# Header bar
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="cm-header">
  <div class="cm-logo">
    <div class="cm-logo-icon">🌿</div>
    <div>
      <div class="cm-logo-text">CropMind AI</div>
      <div class="cm-logo-sub">SMART CROP DIAGNOSTICS</div>
    </div>
  </div>
  <div class="cm-badges">
    <div class="model-pill">🧬 Qwen2.5 + QLoRA</div>
    <div class="model-pill">📷 EfficientNet-B4</div>
    <div class="live-pill">
      <div class="live-dot"></div>
      LIVE
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────
model_area = st.empty()
tokenizer, model, device = None, None, "cpu"
with model_area.container():
    try:
        with st.spinner("🔄  Loading fine-tuned model into memory... (first time ~30s)"):
            tokenizer, model, device = load_model()
        model_area.empty()
    except Exception as e:
        st.error(f"**Model load failed:** `{e}`")
        st.info(
            "**Checklist:**\n"
            "- Adapter exists at `models/llm/qwen2.5_3b_qlora_adapter/`\n"
            "- Run `pip install transformers peft bitsandbytes accelerate`\n"
            "- GPU with ≥8GB VRAM recommended"
        )
        st.stop()
# ─────────────────────────────────────────────────────────────────────────────
# Chat area
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-chat">', unsafe_allow_html=True)
# welcome
if not st.session_state.messages:
    st.markdown(f"""
    <div style="
      max-width:820px;margin:1.5rem auto 0.5rem;
      background:{'linear-gradient(160deg,#081a10 0%,#0a2418 40%,#071218 100%)' if DARK else 'linear-gradient(160deg,#e8f5ee 0%,#d5f5e3 40%,#dbeafe 100%)'};
      border:1px solid {BORDER}33;
      border-radius:22px;
      padding:2.2rem 2.4rem 1.8rem;
      position:relative;overflow:hidden;
    ">
      <div style="position:absolute;top:-30px;right:-20px;font-size:8rem;opacity:0.06;pointer-events:none;">🌿</div>
      <div style="position:absolute;bottom:-20px;left:-10px;font-size:6rem;opacity:0.04;pointer-events:none;">🌾</div>

      <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.6rem;">
        <div style="font-size:2rem;">👋</div>
        <div>
          <div style="font-size:1.5rem;font-weight:800;
            background:linear-gradient(135deg,{ACCENT},{ACCENT2});
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            background-clip:text;">Welcome to CropMind</div>
          <div style="font-size:0.78rem;color:{TEXT_MUT};margin-top:0.15rem;">
            Your AI crop disease expert — trained on 38 disease classes
          </div>
        </div>
      </div>

      <div style="height:1px;background:linear-gradient(90deg,transparent,{BORDER}44,transparent);margin:1rem 0;"></div>

      <div style="font-size:0.68rem;font-weight:700;color:{TEXT_DIM};letter-spacing:0.12em;
        text-transform:uppercase;margin-bottom:0.8rem;">Tell me about your situation</div>

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.7rem;margin-bottom:1.2rem;">
        <div style="background:{'#0d1f1688' if DARK else '#ffffff88'};
          border:1px solid {BORDER}33;border-radius:14px;padding:1rem;
          {'backdrop-filter:blur(8px);' if DARK else ''}transition:border-color 0.2s;">
          <div style="font-size:1.2rem;margin-bottom:0.3rem;">🌱</div>
          <div style="font-size:0.82rem;font-weight:700;color:{ACCENT};margin-bottom:0.2rem;">Crop</div>
          <div style="font-size:0.75rem;color:{TEXT_DIM};font-style:italic;">"I grow tomatoes"</div>
        </div>
        <div style="background:{'#0d1f1688' if DARK else '#ffffff88'};
          border:1px solid {BORDER}33;border-radius:14px;padding:1rem;
          {'backdrop-filter:blur(8px);' if DARK else ''}transition:border-color 0.2s;">
          <div style="font-size:1.2rem;margin-bottom:0.3rem;">🔍</div>
          <div style="font-size:0.82rem;font-weight:700;color:{ACCENT};margin-bottom:0.2rem;">Symptoms</div>
          <div style="font-size:0.75rem;color:{TEXT_DIM};font-style:italic;">"leaves have brown spots"</div>
        </div>
        <div style="background:{'#0d1f1688' if DARK else '#ffffff88'};
          border:1px solid {BORDER}33;border-radius:14px;padding:1rem;
          {'backdrop-filter:blur(8px);' if DARK else ''}transition:border-color 0.2s;">
          <div style="font-size:1.2rem;margin-bottom:0.3rem;">🗺️</div>
          <div style="font-size:0.82rem;font-weight:700;color:{ACCENT};margin-bottom:0.2rem;">Region</div>
          <div style="font-size:0.75rem;color:{TEXT_DIM};font-style:italic;">"I'm in Punjab"</div>
        </div>
        <div style="background:{'#0d1f1688' if DARK else '#ffffff88'};
          border:1px solid {BORDER}33;border-radius:14px;padding:1rem;
          {'backdrop-filter:blur(8px);' if DARK else ''}transition:border-color 0.2s;">
          <div style="font-size:1.2rem;margin-bottom:0.3rem;">📊</div>
          <div style="font-size:0.82rem;font-weight:700;color:{ACCENT};margin-bottom:0.2rem;">Severity</div>
          <div style="font-size:0.75rem;color:{TEXT_DIM};font-style:italic;">"about 40% infected"</div>
        </div>
      </div>

      <div style="display:flex;gap:0.8rem;flex-wrap:wrap;">
        <div style="background:{ACCENT}0d;border:1px solid {ACCENT}25;
          border-radius:10px;padding:0.55rem 0.9rem;
          font-size:0.75rem;color:{ACCENT};display:flex;align-items:center;gap:0.4rem;">
          <span>📷</span> Upload a photo in the sidebar for auto-detection
        </div>
        <div style="background:{ACCENT2}0d;border:1px solid {ACCENT2}25;
          border-radius:10px;padding:0.55rem 0.9rem;
          font-size:0.75rem;color:{ACCENT2};display:flex;align-items:center;gap:0.4rem;">
          <span>💬</span> Type naturally — Hindi-English mix OK!
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
# render history
for msg in st.session_state.messages:
    av = "🌿" if msg["role"] == "assistant" else "👨‍🌾"
    with st.chat_message(msg["role"], avatar=av):
        st.markdown(msg["content"])
st.markdown('</div>', unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────────────────
# Chat input + inference
# ─────────────────────────────────────────────────────────────────────────────
injected = getattr(st.session_state, "_inject", None)
if injected:
    del st.session_state._inject

user_input = st.chat_input(
    "Describe your crop problem... (e.g. 'my potato leaves have dark patches and are wilting')",
    key="chat_in",
) or injected

if user_input and model is not None:
    with st.chat_message("user", avatar="👨‍🌾"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    st.session_state.llm_history.append({"role": "user", "content": user_input})
    st.session_state.msg_count += 1

    history = st.session_state.llm_history

    PLAN_TRIGGERS = [
        "generat", "genarat", "plan", "treatment", "detail", "tell me", "how to",
        "what should", "advise", "suggest", "help me", "cure", "fix", "report", "treat"
    ]
    user_text_history = " ".join(m["content"].lower() for m in history if m["role"] == "user")
    user_explicitly_wants_plan = any(t in user_text_history for t in PLAN_TRIGGERS)

    # Check if user said "yes" in response to assistant asking if they want a plan
    if not user_explicitly_wants_plan and len(history) >= 2:
        last_user_msg = history[-1]["content"].lower()
        last_assistant_msg = history[-2]["content"].lower() if history[-2]["role"] == "assistant" else ""
        
        affirmative_words = ["yes", "yeah", "sure", "ok", "okay", "please", "yep", "do it", "generate"]
        is_affirmative = any(word in last_user_msg.split() for word in affirmative_words)
        assistant_offered_plan = any(word in last_assistant_msg for word in ["plan", "report", "treatment", "detail", "generate"])
        
        if is_affirmative and assistant_offered_plan:
            user_explicitly_wants_plan = True

    use_ft = (
        st.session_state.plan_mode
        or (user_explicitly_wants_plan and ready_for_plan(history))
    )

    if use_ft and not st.session_state.plan_mode:
        st.session_state.plan_mode = True
    history = st.session_state.llm_history

    # always keep the right system prompt at position 0
    if use_ft:
        history[0] = {"role": "system", "content": PLAN_SYSTEM}
    else:
        history[0] = {"role": "system", "content": CHAT_SYSTEM}

    t0 = time.perf_counter()
    full_response = ""

    mode_label = "🧠 Generating treatment report..." if use_ft else "💬 Thinking..."
    with st.chat_message("assistant", avatar="🌿"):
        st.caption(mode_label)
        box = st.empty()
        try:
            for chunk in stream_response(history, tokenizer, model, use_ft=use_ft):
                full_response += chunk
                box.markdown(full_response + "▌")
            box.markdown(full_response)
        except Exception as e:
            full_response = f"⚠️ **Error:** `{e}`"
            box.markdown(full_response)

    elapsed = time.perf_counter() - t0
    st.session_state.resp_times.append(elapsed)
    st.session_state.token_count += int(len(full_response.split()) * 1.3)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.llm_history.append({"role": "assistant", "content": full_response})


    # trim context (keep system + last 20 turns)
    if len(st.session_state.llm_history) > 42:
        st.session_state.llm_history = (
            [st.session_state.llm_history[0]] +
            st.session_state.llm_history[-40:]
        )
    st.rerun()
