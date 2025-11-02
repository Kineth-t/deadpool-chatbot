from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
app = Flask(__name__)

# Load model and tokenizer globally to avoid reloading on every request
model = AutoModelForCausalLM.from_pretrained("./model/deadpool-gpt2")
tokenizer = AutoTokenizer.from_pretrained("./model/deadpool-gpt2")

@app.route('/')
def home():
    return "Hello, Flask!"

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get("input", "")

    if not prompt:
        return jsonify({"error": "No input provided"}), 400

    # Encode input properly
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)

    # Generate output
    outputs = model.generate(
        **inputs,
        max_length=100,
        temperature=0.9,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"response": generated_text})

if __name__ == '__main__':
    app.run(debug=True)