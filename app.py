from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS,cross_origin
import torch
import numpy as np
import base64
import os
import time
from PIL import Image
from io import BytesIO
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import datetime
import csv
torch.manual_seed(1234)

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

def read_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data

def find_string(input_string, word_list):
    for word in word_list:
        if word in input_string:
            return word
    return ""

gender_list = read_csv('gender.csv')
type_list = read_csv('categories.csv')
color_list = read_csv('colors.csv')

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat-Int4", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat-Int4", device_map="auto", trust_remote_code=True).eval()

app = Flask(__name__)
CORS(app)

# Define a route that accepts POST requests
@app.route('/get_caption', methods=['POST'])
@cross_origin()
def process_image():
    start_time = time.time()  # Record the start time

    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create the filename with datetime stamp
    filename = f"{current_datetime}.jpg"

    # Get the base64-encoded image from the request JSON
    request_data = request.get_json()
    image_data = request_data['base64_image']

    # Decode the base64 string
    image_bytes = base64.b64decode(image_data)

    # Create a PIL image object
    img = Image.open(BytesIO(image_bytes))

    # Save the processed image (replace 'p' with the desired file path)
    img.save(filename)

    response, history = model.chat(tokenizer, query = f'<img>{filename}</img>Describe the outfit', history = None)

    modal_name = find_string(response, type_list)
    color = find_string(response, color_list)
    gender = find_string(response, gender_list)

    os.remove(filename)

    end_time = time.time()  # Record the end time
    runtime = end_time - start_time  # Calculate the runtime

    # Return the processed image and caption
    return jsonify({
        'caption': response,
        'modal_name': modal_name,
        'color': color,
        'gender': gender,
        'runtime': runtime
    })

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/')
def index():
    return render_template('index.html')
