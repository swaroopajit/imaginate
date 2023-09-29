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
torch.manual_seed(1234)

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="cuda", trust_remote_code=True).eval()

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

    # Convert image to a numpy array
    # image_array = np.array(img)

    # # Convert numpy array to a tensor
    # image = torch.tensor(image_array, dtype=torch.float32).permute(2, 0, 1) / 255.0
    # image = image.unsqueeze(0).to(device)


    query = tokenizer.from_list_format([
        {'image': filename},
        {'text': 'Describe the outfit'},
    ])
    inputs = tokenizer(query, return_tensors='pt')
    inputs = inputs.to(model.device)
    pred = model.generate(**inputs)
    generated_caption = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)

    # Process the image (e.g., apply filters, resize, etc.)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # # Assuming `model` and `processor` are defined earlier in your code
    # model.to(device)
    # inputs = processor(images=image, return_tensors="pt", do_rescale=False).to(device)
    # pixel_values = inputs.pixel_values
    # generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    # generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    end_time = time.time()  # Record the end time
    runtime = end_time - start_time  # Calculate the runtime


    # Return the processed image and caption
    return jsonify({
        'caption': generated_caption,
        'runtime': runtime
    })

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/')
def index():
    return render_template('index.html')