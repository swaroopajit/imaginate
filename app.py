from flask import Flask, render_template, request
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Initialize the processor and model
processor = AutoProcessor.from_pretrained("swaroopajit/git-base-next")
model = AutoModelForCausalLM.from_pretrained("swaroopajit/git-base-next")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)  # Ensure the model is on the same device as the input

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.get_json()
    encoded_image = data.get('image')
    decoded_image = base64.b64decode(encoded_image.split(',')[1])
    nparr = np.frombuffer(decoded_image, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Save the captured image to a folder
    cv2.imwrite('captures/captured_image.jpg', frame)

    # Convert the image to tensor
    image = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0
    image = image.unsqueeze(0).to(device)

    # Generate caption
    model.to(device)  # Ensure the model is on the same device as the input
    inputs = processor(images=image, return_tensors="pt", do_rescale=False).to(device)
    pixel_values = inputs.pixel_values
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_caption

if __name__ == '__main__':
    app.run(debug=True)
