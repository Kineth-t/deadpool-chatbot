from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
app = Flask(__name__)

# Load model and tokenizer globally to avoid reloading on every request
model = AutoModelForCausalLM.from_pretrained("./model/deadpool-gpt2")
tokenizer = AutoTokenizer.from_pretrained("./model/deadpool-gpt2")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form.get("input") or request.json.get("input")

    if not prompt:
        return jsonify({"error": "No input provided"}), 400

    if not prompt:
        return jsonify({"error": "No input provided"}), 400
    
    print(prompt)

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
    response = generated_text.split("Response:")[-1].strip()

    # If request is from the template form, return to template
    if request.form.get("input"):
        return render_template("index.html", prompt=prompt, response=response)
    else:
        return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)