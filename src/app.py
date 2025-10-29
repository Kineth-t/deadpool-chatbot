from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
app = Flask(__name__)

# Load model and tokenizer globally to avoid reloading on every request
model = AutoModelForCausalLM.from_pretrained("./model/deadpool-gpt2")
tokenizer = AutoTokenizer.from_pretrained("./model/deadpool-gpt2")

@app.route('/')
def home():
    return "Hello, Flask!"

@app.route('/generate')
def generate():
    return "Test"

if __name__ == '__main__':
    app.run(debug=True)