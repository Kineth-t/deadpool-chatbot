from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os
import json
import random

# ── LangChain imports ──────────────────────────────────────────────────────────
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
# 2. LOAD MODEL + ADAPTER
# ──────────────────────────────────────────────
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH    = "./model/deadpool-llama"
adapter_ok      = False

if os.path.isdir(ADAPTER_PATH):
    cfg_path = os.path.join(ADAPTER_PATH, "adapter_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        adapter_base  = cfg.get("base_model_name_or_path", "")
        init_weights  = cfg.get("init_lora_weights", True)

        print(f"\n── Adapter diagnostics ──────────────────────")
        print(f"  base_model        : {adapter_base}")
        print(f"  init_lora_weights : {init_weights}  ← False = properly trained")
        print(f"  r / lora_alpha    : {cfg.get('r')} / {cfg.get('lora_alpha')}")
        print(f"─────────────────────────────────────────────\n")

        base_ok = "TinyLlama" in adapter_base or "tinyllama" in adapter_base.lower()

        if not base_ok:
            print("  ✗ SKIPPING adapter — base model mismatch")
        elif init_weights is True:
            print("  ✗ SKIPPING adapter — init_lora_weights=True means untrained weights.")
            print("    Re-run training with the fixed notebook then replace ./model/deadpool-llama")
        else:
            adapter_ok = True
            print("  ✓ Adapter trained and compatible — will be applied")
else:
    print(f"  ⚠ Adapter not found at {ADAPTER_PATH}")

# Always load tokenizer from base model for correctness
print("\nLoading tokenizer from base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float32,   # float32 required for stable CPU inference
    low_cpu_mem_usage=True,
)

if adapter_ok:
    print("Applying LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("  ✓ LoRA applied")
else:
    print("  Running base model only")
    model = base_model

model.eval()
model = model.to(device)

# ──────────────────────────────────────────────
# 3. GENERATION HELPER
#
# Matches exactly what the fixed notebook's chat() function does:
#   - apply_chat_template for correct token format
#   - slice off prompt tokens before decoding
#   - skip_special_tokens=True for clean output
# ──────────────────────────────────────────────
STOP_STRINGS = ["<|user|>", "<|system|>", "<|assistant|>", "\nUser:", "\nAssistant:"]

def generate_response(prompt: str, max_new_tokens: int = 120, temperature: float = 0.75) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.92,
            top_k=50,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][prompt_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Cut at any role boundary the model might generate
    for stop in STOP_STRINGS:
        if stop in text:
            text = text.split(stop)[0].strip()

    return text


# ──────────────────────────────────────────────
# 4. LANGCHAIN LLM WRAPPER
# Powers ConversationSummaryBufferMemory's auto-summariser.
# Uses low temperature for clean, factual summaries.
# ──────────────────────────────────────────────
class DeadpoolLLM(LLM):

    @property
    def _llm_type(self) -> str:
        return "deadpool-tinyllama"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        return generate_response(prompt, max_new_tokens=100, temperature=0.3)

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs: Any) -> LLMResult:
        generations = [[Generation(text=self._call(p))] for p in prompts]
        return LLMResult(generations=generations)


llm_wrapper = DeadpoolLLM()

# ──────────────────────────────────────────────
# 5. LANGCHAIN MEMORY
# Keeps the last ~400 tokens of conversation verbatim.
# Older turns are automatically summarised to preserve context
# without blowing up the prompt length.
# ──────────────────────────────────────────────
memory = ConversationSummaryBufferMemory(
    llm=llm_wrapper,
    max_token_limit=400,
    human_prefix="User",
    ai_prefix="Deadpool",
    memory_key="chat_history",
    return_messages=False,
)

# ──────────────────────────────────────────────
# 6. SYSTEM PROMPT
# Must match the system prompt used in training exactly.
# ──────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are Deadpool, a.k.a. Wade Wilson — the Merc with a Mouth. "
    "Never identify as an AI or assistant. Be sarcastic, chaotic, "
    "and funny. Break the fourth wall. Roast the user."
)

# ──────────────────────────────────────────────
# 7. ASSISTANT PRIMING PHRASES
#
# Pre-fills the start of the assistant turn so the model continues
# a Deadpool-flavoured sentence instead of defaulting to
# "Hello! I'm a helpful AI assistant!"
# Once the adapter is properly trained this becomes less necessary,
# but it's a cheap safety net that costs nothing.
# ──────────────────────────────────────────────
PRIME_PHRASES = [
    "Oh great, another one.",
    "Ugh, you again.",
    "Well, well, well...",
    "Listen up, pal —",
    "Oh sure, because my day wasn't chaotic enough.",
    "You rang? I was busy bleeding.",
    "Chimichangas. That's all I was thinking about. And now — you.",
    "Fourth wall check: yep, still broken.",
]

# ──────────────────────────────────────────────
# 8. PROMPT BUILDER
#
# Uses apply_chat_template — identical to the notebook's chat() function.
# Prior turns from LangChain memory are injected as real message dicts
# (not a text blob) so the model sees proper multi-turn structure.
# A prime phrase pre-fills the assistant turn to anchor the persona.
# ──────────────────────────────────────────────
def build_prompt(user_input: str, prime: str) -> str:
    history_str = memory.load_memory_variables({}).get("chat_history", "")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Reconstruct prior turns from LangChain memory as proper message dicts
    if history_str.strip():
        for line in history_str.strip().split("\n"):
            if line.startswith("User: "):
                messages.append({"role": "user",      "content": line[len("User: "):]})
            elif line.startswith("Deadpool: "):
                messages.append({"role": "assistant", "content": line[len("Deadpool: "):]})

    messages.append({"role": "user", "content": user_input})

    # apply_chat_template produces the exact token sequence used in training
    base = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,   # appends <|assistant|>\n
    )

    # Prime the assistant turn — model continues FROM this phrase
    return base + prime + " "


# ──────────────────────────────────────────────
# 9. ROUTES
# ──────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    print("\n" + "=" * 50)
    print("[1/6] Request received")

    data = request.get_json() or request.form
    user_input = data.get("input", "").strip()

    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    print(f"[2/6] User input: {user_input}")

    prime = random.choice(PRIME_PHRASES)
    print(f"[3/6] Building prompt (prime: '{prime}')...")
    prompt = build_prompt(user_input, prime)
    print(f"      Prompt:\n{prompt}")

    print("[4/6] Running inference...")
    try:
        raw_continuation = generate_response(prompt)
        print(f"      Raw continuation: {raw_continuation}")
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        return jsonify({"error": str(e)}), 500

    # Prepend the prime phrase — generate_response already stripped the prompt
    response = (prime + " " + raw_continuation).strip()
    print(f"[5/6] Final response: {response}")

    if not response or response.strip() == prime.strip():
        response = f"{prime} Not gonna lie, I've got nothing. Which is saying something for a guy who never shuts up."
        print("      [WARN] Thin response — using fallback")

    print("[6/6] Saving to LangChain memory...")
    memory.save_context({"input": user_input}, {"output": response})

    print(f"\nDeadpool: {response}")
    print("=" * 50 + "\n")

    if request.form.get("input"):
        return render_template("index.html", prompt=user_input, response=response)
    return jsonify({"response": response})


@app.route("/reset", methods=["POST"])
def reset():
    memory.clear()
    return jsonify({"status": "conversation reset"})


# ──────────────────────────────────────────────
# 10. RUN
# Run with: PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python app.py
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False)