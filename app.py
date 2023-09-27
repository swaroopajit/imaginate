from flask import Flask, render_template, request, jsonify, Response
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import numpy as np
import base64
import os
import time
from PIL import Image
from io import BytesIO

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

global capture
capture=0

app = Flask(__name__)

# Initialize the processor and model
processor = AutoProcessor.from_pretrained("swaroopajit/git-base-next-refined")
model = AutoModelForCausalLM.from_pretrained("swaroopajit/git-base-next-refined")

app = Flask(__name__)

# Define a route that accepts POST requests
@app.route('/get_caption', methods=['POST'])
def process_image():
    start_time = time.time()  # Record the start time

    # Get the base64-encoded image from the request JSON
    request_data = request.get_json()
    image_data = request_data['base64_image']

    # Decode the base64 string
    image_bytes = base64.b64decode(image_data)

    # Create a PIL image object
    img = Image.open(BytesIO(image_bytes))

    # Process the image (e.g., apply filters, resize, etc.)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert image to a numpy array
    image_array = np.array(img)

    # Convert numpy array to a tensor
    image = torch.tensor(image_array, dtype=torch.float32).permute(2, 0, 1) / 255.0
    image = image.unsqueeze(0).to(device)

    # Assuming `model` and `processor` are defined earlier in your code
    model.to(device)
    inputs = processor(images=image, return_tensors="pt", do_rescale=False).to(device)
    pixel_values = inputs.pixel_values
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    end_time = time.time()  # Record the end time
    runtime = end_time - start_time  # Calculate the runtime

    # Save the processed image (replace 'p' with the desired file path)
    img.save('processed_image.jpg')

    # Return the processed image and caption
    return jsonify({
        'caption': generated_caption,
        'runtime': runtime
    })

if __name__ == '__main__':
    app.run(debug=True)


# def generate_frames():
#     camera = cv2.VideoCapture(0)
#     global capture
#     while True:
#         success, frame = camera.read()
#         if not success:
#             break
#         else:
#             if(capture):
#                 # do the infrencing here hopefully
#                 start_time = time.time()  # Record the start time
#                 capture=0
#                 now = datetime.datetime.now()
#                 p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
#                 cv2.imwrite(p, frame)

#                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
#                 # Convert the image to tensor
#                 image = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0

#                 model.to(device)  # Ensure the model is on the same device as the input
#                 inputs = processor(images=image, return_tensors="pt", do_rescale=False).to(device)
#                 pixel_values = inputs.pixel_values
#                 generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
#                 generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

#                 end_time = time.time()  # Record the end time
#                 runtime = end_time - start_time  # Calculate the runtime

#                 print(generated_caption)
#                 print(runtime)

#             try:
#                 ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
#                 frame = buffer.tobytes()
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#             except Exception as e:
#                 pass
#     camera.release()
#     cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/capture',methods=['POST'])
# def capture():
#     global capture
#     capture=1
#     return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
