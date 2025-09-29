# Creating a connection between model and simulator

# for webserver
# For real time connection between client and server
import socketio
# For event listening socket
import eventlet
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
# Flask - Python micro framework for initilizing webapp
from flask import Flask
from keras.models import load_model

# Creating a socketio web server
# Requires a middleware for trafficing data to it
# Flask used for it
sio = socketio.Server()
# replaces main
app = Flask(__name__)  

speed_limit = 10

# Function to send control back to simulator
def send_control(steering_angle, throttle):
    print('Simulator connected')
    sio.emit('steer', data={
        'steering_angle' : str(steering_angle),
        'throttle' : str(throttle)
    })

# Preprocessing data
def image_preprocess(img):
    # Cropping the top and bottom of image with unnecessary data
    # Decided based on viewing the data
    img = img[60:135, :, : ]
    # Nvidia model is used
    # YUV color space required for handling
    # Y - Brightness, UV - Cromiance (adds colors)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # Adding Gaussian blur
    # Smoothens image
    img = cv2.GaussianBlur(img, (3,3), 0)
    # Resize image for faster handling
    # Input size of Nvidia architecture (for consistency)
    img = cv2.resize(img, (200, 66))
    # Normalization
    # No visual impact
    img = img/255
    return img

#Identifying the connection
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)

# Listen for updates send to telemetry from simulator
@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    # Decoding the image
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    # Changing image as array
    image = np.asarray(image)
    # Preprocessing the image
    image = image_preprocess(image)
    # changing to 4d array which is expected
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed/speed_limit
    print(f'Steering angle : {steering_angle}, \nThrottle: {throttle},\nSpeed: {speed}')
    send_control(steering_angle, throttle)

if __name__ == '__main__':
    model = load_model('model_tf.keras')
    # Setting flask app as middleware
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
