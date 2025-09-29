# Creating a connection between model and simulator
# Flask - Python micro framework for initilizing webapp
from flask import Flask

# replaces main
app = Flask(__name__)  

# Router decorator
# If user routes to this location in browser, it will displays welcome
@app.route('/home')
def greeting():
    return 'Welcome !'

if __name__ == '__main__':
    app.run(port=3000)