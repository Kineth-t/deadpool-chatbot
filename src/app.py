from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from peft import PeftModel

import torch

app = Flask(__name__)

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# 1. LOAD MODEL
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    low_cpu_mem_usage=True
)

model = PeftModel.from_pretrained(
    base_model,
    "./model/deadpool-lora"
)

tokenizer = AutoTokenizer.from_pretrained("./model/deadpool-lora")
tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=120,
    temperature=0.9,
    top_p=0.9,
    repetition_penalty=1.2,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=pipe)


# 2. MEMORY (AUTO MANAGED)
memory = ConversationSummaryMemory(
    llm=llm,
    return_messages=False
)


# 3. DEADPOOL PROMPT
template = """
You are Deadpool (Wade Wilson).

You are sarcastic, chaotic, funny, and break the fourth wall.
You mock the user and make ridiculous jokes.

Conversation so far:
{history}

User: {input}
Deadpool:
"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template
)


# 4. CONVERSATION CHAIN
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=False
)


# ROUTES
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json() or request.form
    user_input = data.get("input")

    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    print(f"User: {user_input}")

    # Generate response via LangChain
    response = conversation.predict(input=user_input).strip()

    print(f"Deadpool: {response}")

    # Return response
    if request.form.get("input"):
        return render_template("index.html", prompt=user_input, response=response)
    else:
        return jsonify({"response": response})


# RUN
if __name__ == '__main__':
    app.run(debug=True)