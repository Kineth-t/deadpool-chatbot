from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import random

app = Flask(__name__)

# Load model and tokenizer once
model = AutoModelForCausalLM.from_pretrained("./model/deadpool-gpt2")
tokenizer = AutoTokenizer.from_pretrained("./model/deadpool-gpt2")

# Global conversation memory
conversation_history = ""

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
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    global conversation_history

    prompt = request.form.get("input") or request.json.get("input")

    if not prompt:
        return jsonify({"error": "No input provided"}), 400

    print(f"User: {prompt}")

    # Append user input
    conversation_history += f"\n### User:\n{prompt}\n### Assistant:\n"

    # Trim history
    conversation_history = trim_history(conversation_history)

    # Tokenize full conversation
    inputs = tokenizer(conversation_history, return_tensors="pt", padding=True)

    # Generate continuation
    outputs = model.generate(
        **inputs,
        max_length=inputs["input_ids"].shape[1] + 100,
        temperature=0.9,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract ONLY the new response
    new_response = generated_text[len(conversation_history):].strip()

    # Clean + fix sentence ending
    new_response = clean_response(new_response)
    new_response = ensure_sentence_end(new_response)

    print(f"Bot: {new_response}")

    # Append assistant response
    conversation_history += f"{new_response}\n"

    # Return response
    if request.form.get("input"):
        return render_template("index.html", prompt=prompt, response=new_response)
    else:
        return jsonify({"response": new_response})

# Run App
if __name__ == '__main__':
    app.run(debug=True)