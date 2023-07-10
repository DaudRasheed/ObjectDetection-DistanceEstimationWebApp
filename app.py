import cv2
import numpy as np
from flask import Flask, render_template, Response

# Known parameters
KNOWN_WIDTH = 0.2  # Width of the object in meters
FOCAL_LENGTH = 700  # Focal length of the camera in pixels

# Load the model and labels
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
file_name = 'Labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

# Set model input properties
model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

app = Flask(__name__,template_folder='Tamplates')

def detect_objects():
    # Open the video capture
    cap = cv2.VideoCapture(0)  # Change index if using a different camera
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)

        if len(ClassIndex) != 0:
            for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
                if ClassInd <= 80:
                    # Draw bounding box and label
                    cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                    cv2.putText(frame, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), cv2.FONT_HERSHEY_PLAIN,
                                fontScale=2, color=(0, 255, 0), thickness=3)

                    # Distance estimation
                    object_width = boxes[2] - boxes[0]  # Width of the object in pixels

                    # Check for negative object width
                    if object_width > 0:
                        distance = (KNOWN_WIDTH * FOCAL_LENGTH) / object_width  # Estimate distance using similar triangles

                        # Display distance
                        text = f"Distance: {distance:.2f} meters"
                    else:
                        text = "Distance: 0"

                    cv2.putText(frame, text, (boxes[0]+10, boxes[1]+80), cv2.FONT_HERSHEY_PLAIN,
                                fontScale=2, color=(0, 255, 0), thickness=2)

        # Convert the frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    # Release the video capture
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_objects(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
