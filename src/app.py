from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os
import json
import re

from langchain.memory import ConversationSummaryBufferMemory
from langchain.llms.base import LLM
from langchain.schema import LLMResult, Generation
from typing import Any, List, Optional

app = Flask(__name__)

# ──────────────────────────────────────────────
# 1. DEVICE
# ──────────────────────────────────────────────
device = torch.device("cpu")
print(f"Using device: {device}")

# ──────────────────────────────────────────────
# 2. TOKENIZER
# ──────────────────────────────────────────────
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH    = "./model/deadpool-llama"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# ──────────────────────────────────────────────
# 3. BASE MODEL (summarizer)
# ──────────────────────────────────────────────
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME, dtype=torch.float32, low_cpu_mem_usage=True,
)
base_model.eval()
base_model = base_model.to(device)

# ──────────────────────────────────────────────
# 4. DEADPOOL MODEL (base + LoRA)
# ──────────────────────────────────────────────
adapter_ok = False

if os.path.isdir(ADAPTER_PATH):
    cfg_path = os.path.join(ADAPTER_PATH, "adapter_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        adapter_base     = cfg.get("base_model_name_or_path", "")
        init_weights     = cfg.get("init_lora_weights", True)
        safetensors_path = os.path.join(ADAPTER_PATH, "adapter_model.safetensors")
        size_mb          = os.path.getsize(safetensors_path) / 1e6 if os.path.exists(safetensors_path) else 0

        print(f"\n── Adapter diagnostics ──────────────────────")
        print(f"  base_model        : {adapter_base}")
        print(f"  init_lora_weights : {init_weights}")
        print(f"  adapter size      : {size_mb:.1f} MB")
        print(f"─────────────────────────────────────────────\n")

        base_ok = "TinyLlama" in adapter_base or "tinyllama" in adapter_base.lower()
        trained  = (not init_weights) or (size_mb > 1.0)

        if not base_ok:
            print("SKIPPING — base model mismatch")
        elif not trained:
            print("SKIPPING — adapter appears untrained")
        else:
            adapter_ok = True
            print("Adapter ready")
else:
    print(f"Adapter not found at {ADAPTER_PATH}")

if adapter_ok:
    print("Applying LoRA adapter...")
    deadpool_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    deadpool_model.eval()
    deadpool_model = deadpool_model.to(device)
    print("Deadpool model ready")
else:
    print("Using base model only")
    deadpool_model = base_model

print("Models loaded!\n")

# ──────────────────────────────────────────────
# 5. SANITIZER
# ──────────────────────────────────────────────
SPECIAL_TOKEN_RE = re.compile(r'<\|[^|]+\|>|</s>|<s>')
JUNK_RE          = re.compile(r'[®™©♥♦♠♣♤♡♢♧]|[\U0001F300-\U0001F9FF]|\n{2,}', re.UNICODE)

def sanitize(text: str) -> str:
    text = SPECIAL_TOKEN_RE.sub("", text)
    text = JUNK_RE.sub(" ", text)
    text = re.split(r'https?://\S+|www\.\S+|\S+\.com\S*', text)[0]
    return " ".join(text.split()).strip()

# ──────────────────────────────────────────────
# 6. GENERATION
# ──────────────────────────────────────────────
STOP_STRINGS = [
    "<|user|>", "<|system|>", "<|assistant|>",
    "\nUser:", "\nAssistant:",
]

def _decode(model, prompt: str, max_new_tokens: int, temperature: float,
            min_new_tokens: int = 10) -> str:
    inputs     = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.90,
            top_k=50,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
    for stop in STOP_STRINGS:
        if stop in text:
            text = text.split(stop)[0].strip()
    return text


def deadpool_generate(prompt: str) -> str:
    text = _decode(
        deadpool_model, prompt,
        max_new_tokens=120,
        temperature=0.70,
        min_new_tokens=10,
    )
    return sanitize(text)


def summarizer_generate(prompt: str) -> str:
    return _decode(base_model, prompt, max_new_tokens=120, temperature=0.3, min_new_tokens=10)

# ──────────────────────────────────────────────
# 7. LANGCHAIN WRAPPER (summarizer only)
# ──────────────────────────────────────────────
class SummarizerLLM(LLM):

    @property
    def _llm_type(self) -> str:
        return "tinyllama-summarizer"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        return summarizer_generate(prompt)

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None,
                  **kwargs: Any) -> LLMResult:
        generations = [[Generation(text=self._call(p))] for p in prompts]
        return LLMResult(generations=generations)


summarizer_llm = SummarizerLLM()

# ──────────────────────────────────────────────
# 8. MEMORY
# ──────────────────────────────────────────────
conversation_history: list = []
MAX_RECENT_TURNS = 6

memory = ConversationSummaryBufferMemory(
    llm=summarizer_llm,
    max_token_limit=300,
    human_prefix="User",
    ai_prefix="Deadpool",
    memory_key="chat_history",
    return_messages=False,
)

# ──────────────────────────────────────────────
# 9. SYSTEM PROMPT
# ──────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are Deadpool, a.k.a. Wade Wilson — the Merc with a Mouth. "
    "You are NOT an AI, NOT a chatbot, NOT an assistant. You ARE Wade Wilson. "
    "Be chaotic, funny, and break the fourth wall. "
    "Keep responses short and punchy. Never ramble."
)

# ──────────────────────────────────────────────
# 10. PROMPT BUILDER
# ──────────────────────────────────────────────
def build_prompt(user_input: str) -> str:
    system_block = SYSTEM_PROMPT

    if len(conversation_history) > MAX_RECENT_TURNS:
        summary = memory.load_memory_variables({}).get("chat_history", "").strip()
        if summary:
            system_block += f"\n\n[Earlier in this conversation:]\n{summary}"

    messages = [{"role": "system", "content": system_block}]

    for turn in conversation_history[-MAX_RECENT_TURNS:]:
        messages.append({"role": "user",      "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["deadpool"]})

    messages.append({"role": "user", "content": user_input})

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

# ──────────────────────────────────────────────
# 11. ROUTES
# ──────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():

    data = request.get_json() or request.form
    user_input = data.get("input", "").strip()
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    print(f"User: {user_input}")

    prompt = build_prompt(user_input)
    print(f"Prompt:\n{prompt}")

    try:
        continuation = deadpool_generate(prompt)
        print(f"Continuation: {continuation}")
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500

    response = continuation.strip()
    if not continuation:
        response = "...and that's all I got. Shocking, I know."

    clean_resp  = sanitize(response)
    clean_input = sanitize(user_input)

    conversation_history.append({"user": clean_input, "deadpool": clean_resp})
    memory.save_context({"input": clean_input}, {"output": clean_resp})

    print(f"\nDeadpool: {response}\n")

    if request.form.get("input"):
        return render_template("index.html", prompt=user_input, response=response)
    return jsonify({"response": response})


@app.route("/reset", methods=["POST"])
def reset():
    conversation_history.clear()
    memory.clear()
    return jsonify({"status": "conversation reset"})


if __name__ == "__main__":
    app.run(debug=False)