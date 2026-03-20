from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import random

app = Flask(__name__)

# Load model and tokenizer once
model = AutoModelForCausalLM.from_pretrained("./model/deadpool-gpt2")
tokenizer = AutoTokenizer.from_pretrained("./model/deadpool-gpt2")

# Global conversation memory
conversation = []

# Limit history size to prevent overflow
MAX_HISTORY_CHARS = 2000


# Helper Functions
def clean_response(text):
    """Extract last complete sentence"""
    text = text.strip()
    match = re.search(r'(.+[.!?])', text)
    return match.group(1) if match else text


def ensure_sentence_end(text):
    """Ensure response ends with proper punctuation"""
    text = text.strip()

    if not text:
        return text

    if text[-1] not in {'.', '!', '?'}:
        text += random.choice(['.', '!', '?'])

    return text


def trim_history(history):
    """Keep only recent part of conversation"""
    return history[-MAX_HISTORY_CHARS:]



# Routes
@app.route('/')
def home():
    return render_template('index.html', history=conversation)


@app.route('/generate', methods=['POST'])
def generate():
    global conversation

    prompt = request.form.get("input") or request.json.get("input")

    if not prompt:
        return jsonify({"error": "No input provided"}), 400

    print(f"User: {prompt}")

    # Add user message
    conversation.append({"role": "user", "content": prompt})

    # -------------------------
    # Build prompt from history
    # -------------------------
    full_prompt = ""
    for msg in conversation:
        if msg["role"] == "user":
            full_prompt += f"User: {msg['content']}\n"
        else:
            full_prompt += f"Bot: {msg['content']}\n"

    full_prompt += "Bot:"

    # Trim prompt (prevent overflow)
    full_prompt = full_prompt[-MAX_HISTORY_CHARS:]

    # Tokenize
    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True)

    # Generate
    outputs = model.generate(
        **inputs,
        max_length=inputs["input_ids"].shape[1] + 100,
        temperature=0.9,
        top_p=0.98,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only new response
    new_response = generated_text[len(full_prompt):].strip()

    # Clean + fix
    new_response = clean_response(new_response)
    new_response = ensure_sentence_end(new_response)

    print(f"Bot: {new_response}")

    # Save bot response
    conversation.append({"role": "bot", "content": new_response})

    # Return JSON (for chat UI)
    return jsonify({"response": new_response})

# Run App
if __name__ == '__main__':
    app.run(debug=True)