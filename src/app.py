from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch

app = Flask(__name__)

# ──────────────────────────────────────────────
# 1. DEVICE
# Run with: PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python app.py
# ──────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ──────────────────────────────────────────────
# 2. LOAD MODEL
# ──────────────────────────────────────────────
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

print("Loading base model on CPU...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
)

print("Applying LoRA adapter...")
model = PeftModel.from_pretrained(base_model, "./model/deadpool-lora")
model.eval()

print(f"Moving model to {device}...")
model = model.to(device)

# ──────────────────────────────────────────────
# 3. TOKENIZER
# ──────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# ──────────────────────────────────────────────
# 4. PIPELINE
# ──────────────────────────────────────────────
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    temperature=0.9,
    top_p=0.95,
    repetition_penalty=1.1,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    device=device,
)

print("Ready!\n")

# ──────────────────────────────────────────────
# 5. MEMORY STATE
#
# This mimics LangChain's ConversationSummaryMemory:
#   - Keep the last MAX_RECENT turns verbatim (for precise context)
#   - Once history exceeds MAX_BEFORE_SUMMARY turns, summarize
#     the oldest ones into `running_summary` and discard them
#   - Every prompt = summary (if any) + recent turns + new input
#
# Example after 10 turns with MAX_RECENT=4, MAX_BEFORE_SUMMARY=6:
#   summary  = "User asked about Deadpool's past. Deadpool was sarcastic..."
#   recent   = [turn7, turn8, turn9, turn10]
# ──────────────────────────────────────────────
conversation_history = []   # list of {"user": ..., "bot": ...}
running_summary = ""        # compressed summary of old turns
MAX_RECENT = 4              # how many recent turns to keep verbatim
MAX_BEFORE_SUMMARY = 6      # trigger summarization after this many turns


# ──────────────────────────────────────────────
# 6. SUMMARIZER
# Calls the model with a neutral summarization prompt (no Deadpool
# persona) so the summary is clean and factual, not chaotic.
# ──────────────────────────────────────────────
def summarize_turns(turns: list, existing_summary: str) -> str:
    turns_text = ""
    for turn in turns:
        turns_text += f"User: {turn['user']}\nDeadpool: {turn['bot']}\n"

    if existing_summary:
        context = f"Existing summary:\n{existing_summary}\n\nNew turns to add:\n{turns_text}"
    else:
        context = f"Conversation:\n{turns_text}"

    prompt = (
        f"<s>[INST] Summarize the following conversation in 2-3 sentences. "
        f"Be concise and factual. Only output the summary, nothing else.\n\n"
        f"{context}\n[/INST]\n"
    )

    raw = pipe(prompt, max_new_tokens=100, temperature=0.3)[0]["generated_text"]

    # Strip the prompt from output
    if raw.startswith(prompt):
        raw = raw[len(prompt):]

    return raw.strip()


# ──────────────────────────────────────────────
# 7. MEMORY MANAGER
# Called after every turn. If history is too long,
# summarize the oldest turns and compress them.
# ──────────────────────────────────────────────
def maybe_summarize():
    global running_summary, conversation_history

    if len(conversation_history) > MAX_BEFORE_SUMMARY:
        # Split: summarize the old turns, keep the recent ones verbatim
        turns_to_summarize = conversation_history[:-MAX_RECENT]
        conversation_history = conversation_history[-MAX_RECENT:]

        print("Summarizing old turns...")
        running_summary = summarize_turns(turns_to_summarize, running_summary)
        print(f"Summary: {running_summary}\n")


# ──────────────────────────────────────────────
# 8. PROMPT BUILDER
# ──────────────────────────────────────────────
def build_prompt(user_input: str) -> str:
    # Start with summary of old context (if any)
    memory_str = ""
    if running_summary:
        memory_str += f"[Earlier in the conversation: {running_summary}]\n\n"

    # Append recent turns verbatim
    for turn in conversation_history[-MAX_RECENT:]:
        memory_str += f"User: {turn['user']}\nDeadpool: {turn['bot']}\n"

    return (
        "<s>[INST] You are Deadpool (Wade Wilson). "
        "Be sarcastic, chaotic, funny, and break the fourth wall. "
        "Mock the user and make ridiculous jokes.\n\n"
        f"{memory_str}"
        f"User: {user_input}\n[/INST]\n"
    )


# ──────────────────────────────────────────────
# 9. RESPONSE CLEANER
# ──────────────────────────────────────────────
def clean_response(raw: str, prompt: str) -> str:
    if raw.startswith(prompt):
        raw = raw[len(prompt):]
    for stop in ["[INST]", "User:", "\nUser", "</s>"]:
        if stop in raw:
            raw = raw.split(stop)[0]
    return raw.strip()


# ──────────────────────────────────────────────
# 10. ROUTES
# ──────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    print("\n" + "="*50)
    print("[1/7] Request received")

    data = request.get_json() or request.form
    user_input = data.get("input", "").strip()

    if not user_input:
        print("[ERROR] No input provided")
        return jsonify({"error": "No input provided"}), 400

    print(f"[2/7] User input: {user_input}")
    print(f"      History length: {len(conversation_history)} turns")
    print(f"      Running summary: {'yes' if running_summary else 'none'}")

    print("[3/7] Building prompt...")
    prompt = build_prompt(user_input)
    print(f"      Prompt length: {len(prompt)} chars")
    print(f"      Prompt preview: {prompt[:200].strip()}...")

    print("[4/7] Running inference (this is the slow part)...")
    try:
        raw = pipe(prompt)[0]["generated_text"]
        print(f"      Raw output length: {len(raw)} chars")
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        return jsonify({"error": str(e)}), 500

    print("[5/7] Cleaning response...")
    response = clean_response(raw, prompt)
    print(f"      Cleaned response: {response}")

    print("[6/7] Saving to history...")
    conversation_history.append({"user": user_input, "bot": response})
    print(f"      History is now {len(conversation_history)} turns")

    print("[7/7] Checking if summarization needed...")
    maybe_summarize()

    print(f"\nDeadpool: {response}")
    print("="*50 + "\n")

    if request.form.get("input"):
        return render_template("index.html", prompt=user_input, response=response)
    else:
        return jsonify({"response": response})


@app.route("/reset", methods=["POST"])
def reset():
    global running_summary
    conversation_history.clear()
    running_summary = ""
    return jsonify({"status": "conversation reset"})


# ──────────────────────────────────────────────
# 11. RUN
# Always run with:
#   PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python app.py
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False)