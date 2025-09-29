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

@sio.on('connect') #message, disconnect
def connect(sid, environ):
    print('Connected')

if __name__ == '__main__':
    # Setting flask app as middleware
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
