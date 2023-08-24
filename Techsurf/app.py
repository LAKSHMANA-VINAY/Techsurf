from flask import Flask, render_template, request
from transformers import pipeline
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf

app = Flask(__name__)

summerizer=pipeline('summarization')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)


def summerize_text(inp):
    summary=summerizer(inp,max_length=100,min_length=15,do_sample=False)[0]['summary_text']
    return summary

def generate_text(inp):
    input_ids = tokenizer.encode(inp, return_tensors="tf")
    beam_output = model.generate(input_ids, max_length=500, num_beams=3, do_sample=True, no_repeat_ngram_size=5, top_k=50, temperature=0.8, early_stopping=True)
    output = tokenizer.decode(beam_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return ".".join(output.split(".")[:-1]) + "."

@app.route("/", methods=["GET", "POST"])
def index():
    output_text = ""
    if request.method == "POST":
        input_text = request.form.get("input_text")
        summary=summerize_text(input_text)
        full_content = input_text+"\n"+summary
        output_text = generate_text(full_content)
    return render_template("index.html", output_text=output_text)

if __name__ == "__main__":
    app.run(debug=True)
