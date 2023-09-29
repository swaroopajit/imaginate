from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS,cross_origin
import torch
import numpy as np
import base64
import os
import time
from PIL import Image
from io import BytesIO

from argparse import ArgumentParser
from pathlib import Path

import copy
import os
import re
import secrets
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, GenerationConfig
)
from huggingface_hub import snapshot_download


DEFAULT_CKPT_PATH = '4bit/Qwen-VL-Chat-Int4'
REVISION = 'v1.0.0'
BOX_TAG_PATTERN = r"<box>([\s\S]*?)</box>"
PUNCTUATION = "！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."

app = Flask(__name__)
CORS(app)

def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--revision", type=str, default=REVISION)
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")

    parser.add_argument("--share", action="store_true", default=False,
                        help="Create a publicly shareable link for the interface.")
    parser.add_argument("--inbrowser", action="store_true", default=False,
                        help="Automatically launch the interface in a new tab on the default browser.")
    parser.add_argument("--server-port", type=int, default=8000,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="127.0.0.1",
                        help="Demo server name.")

    args = parser.parse_args()
    return args

def _load_model_tokenizer(args):
    model_id = args.checkpoint_path
    model_dir = snapshot_download(model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, resume_download=True,
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map=device_map,
        trust_remote_code=True,
        resume_download=True,
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        model_dir, trust_remote_code=True, resume_download=True,
    )

    return model, tokenizer

# MARK :- don't change this
def _parse_text(text):
    print('parse_text called')
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text

def predict(model, tokenizer, image_b64, query):
    print("User: " + query)
    history = [query]  # Assuming you want to include the query in the conversation history
    message = image_b64  # Assuming image_b64 contains the base64 encoded image

    response, _ = model.chat(tokenizer, message, history=history)
    # Process the response if needed
    return response

def add_file(history, task_history, file):
    print('add_file called')
    history = history + [((file.name,), None)]
    task_history = task_history + [((file.name,), None)]
    return history, task_history

def reset_state(task_history):
    task_history.clear()
    return []


# Define a route that accepts POST requests
@app.route('/get_caption', methods=['POST'])
@cross_origin()
def process_image():
    start_time = time.time()  # Record the start time

    # # Get the base64-encoded image from the request JSON
    request_data = request.get_json()
    image_data = request_data['base64_image']

    # # Decode the base64 string
    # image_bytes = base64.b64decode(image_data)

    # # Create a PIL image object
    # img = Image.open(BytesIO(image_bytes))

    # # Process the image (e.g., apply filters, resize, etc.)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Convert image to a numpy array
    # image_array = np.array(img)

    # # Convert numpy array to a tensor
    # image = torch.tensor(image_array, dtype=torch.float32).permute(2, 0, 1) / 255.0
    # image = image.unsqueeze(0).to(device)

    # # Assuming `model` and `processor` are defined earlier in your code
    # model.to(device)
    # inputs = processor(images=image, return_tensors="pt", do_rescale=False).to(device)
    # pixel_values = inputs.pixel_values
    # generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    # generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    args = _get_args()
    model, tokenizer = _load_model_tokenizer(args)

    query = 'describe the outfit'
    generated_caption = predict(model, tokenizer, image_data, query)

    # Process the response if needed
    
    end_time = time.time()  # Record the end time
    runtime = end_time - start_time  # Calculate the runtime

    # Save the processed image (replace 'p' with the desired file path)
    # img.save('processed_image.jpg')

    # Return the processed image and caption
    return jsonify({
        'caption': generated_caption,
        'runtime': runtime
    })

# def process_image_new():


if __name__ == '__main__':
    app.run(debug=True)



if __name__ == '__main__':
    app.run(debug=True)
