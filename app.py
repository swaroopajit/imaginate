from flask import Flask, render_template, request, jsonify, Response
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import cv2
import numpy as np
import base64
import time

app = Flask(__name__)

# Initialize the processor and model
processor = AutoProcessor.from_pretrained("swaroopajit/git-base-next-refined")
model = AutoModelForCausalLM.from_pretrained("swaroopajit/git-base-next-refined")

# Initialize variable to store captured frame
captured_frame = None

def generate_frames():
    cap = cv2.VideoCapture(0)  # 0 indicates the default camera (usually the webcam)

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    start_time = time.time()  # Record the start time

    data = request.get_json()
    encoded_image = data.get('image')
    decoded_image = base64.b64decode(encoded_image.split(',')[1])
    nparr = np.frombuffer(decoded_image, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Save the captured image to a folder
    cv2.imwrite('captures/captured_image.jpg', frame)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert the image to tensor
    image = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0
    image = image.unsqueeze(0).to(device)

    print(device)
    model.to(device)  # Ensure the model is on the same device as the input
    inputs = processor(images=image, return_tensors="pt", do_rescale=False).to(device)
    pixel_values = inputs.pixel_values
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    end_time = time.time()  # Record the end time
    runtime = end_time - start_time  # Calculate the runtime

    return jsonify({'caption': generated_caption, 'runtime': runtime})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    global captured_frame
    captured_frame = cv2.imdecode(np.frombuffer(request.files['image'].read(), np.uint8), -1)
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)
