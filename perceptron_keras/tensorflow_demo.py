'''
Things done:
1. Randomly plotted some data
2. Trained a model using the data by creating a decision boundary
3. Based on the model, predict future inputs
'''

# %%
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
#Importing instead of gradient algorithm since it is expensive
from keras.optimizers import Adam  

#%%
n_pts = 500
np.random.seed(0)
Xa = np.array([np.random.normal(13, 2, n_pts),
               np.random.normal(12, 2, n_pts)]).T
Xb = np.array([np.random.normal(8, 2, n_pts),
               np.random.normal(6, 2, n_pts)]).T
 
X = np.vstack((Xa, Xb))
Y = np.matrix(np.append(np.zeros(n_pts), np.ones(n_pts))).T
 
plt.scatter(X[:n_pts,0], X[:n_pts,1])
plt.scatter(X[n_pts:,0], X[n_pts:,1])

# %%
model = Sequential()
model.add(Dense(units=1, input_shape=(2,), activation='sigmoid' ))
#Adding optimizer
adam = Adam(learning_rate = 0.1)
#Set learning process
model.compile(adam, loss='binary_crossentropy', metrics=['accuracy'])
#Train model to fit to our data
h = model.fit(x=X, y=Y, verbose=1, batch_size=50, epochs=500, shuffle='true')
# %%
plt.plot(h.history['accuracy'])
plt.title('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy'])
# %%
plt.plot(h.history['loss'])
plt.title('loss')
plt.xlabel('epoch')
plt.legend(['loss'])
# %%
#Plot the result
# -1 and +1 for tolerance - makes data more spacious
def plot_decision_boundary(X, Y, model):
    x_span = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1)
    y_span = np.linspace(min(X[:, 1]) - 1, max(X[:, 1]) + 1)
    # square 2d array in numpy mesh grid func
    # returns 2 2d 50*50 matrix for each span 
    xx, yy = np.meshgrid(x_span, y_span)
    # converting to 1d because currently every y coordinate has 50 x coordinates
    # so converting to columnwise to concantenate
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    # returns array of predictions with probability of returning 1 or 0
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape)
    # plotting contour 
    plt.contourf(xx, yy, z)
    plt.scatter(X[:n_pts,0], X[:n_pts,1])
    plt.scatter(X[n_pts:,0], X[n_pts:,1])

    # Test input and predicting
    x = 7.5
    y = 5
    point = np.array([[x, y]])
    prediction = model.predict(point)
    plt.plot([x], [y], marker = "o", markersize = 10, color = "red")
    print ("Prediction is : ", prediction)    

# %%
plot_decision_boundary(X, Y, model)
# %%
