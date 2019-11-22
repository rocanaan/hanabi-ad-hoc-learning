import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(units = 512, activation = 'relu', input_dim = 100))
model.add(Dense(units = 512, activation = 'relu'))
model.add(Dense(units = 20, activation = 'softmax'))


