# Creating a connection between model and simulator

# for webserver
# For real time connection between client and server
import socketio
# For event listening socket
import eventlet
# Flask - Python micro framework for initilizing webapp
from flask import Flask

# Creating a socketio web server
# Requires a middleware for trafficing data to it
# Flask used for it
sio = socketio.Server()
# replaces main
app = Flask(__name__)  

# Function to send control back to simulator
def send_control(steering_angle, throttle):
    print('Simulator connected')
    sio.emit('steer', data={
        'steering_angle' : str(steering_angle),
        'throttle' : str(throttle)
    })

#Identifying the connection
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 1)

if __name__ == '__main__':
    # Setting flask app as middleware
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
