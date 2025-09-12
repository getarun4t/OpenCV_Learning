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
