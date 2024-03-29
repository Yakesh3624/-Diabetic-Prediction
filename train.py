from numpy import loadtxt #for loading csv file
import numpy
from keras.models import Sequential #simple neural network
from keras.layers import Dense 
from keras.models import model_from_json #json files are the files which store data structure simply it contain the structure of neural nertork


dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
x = dataset[:757,0:8]
y = dataset[:757,8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
model.fit(x, y, epochs=5, batch_size=10)

#compile is just configuring the network
#the fit does the training including forward pass and by calculating the error it does the back pass(pack propagation)


# optimisers are the functions which is used to find the weight with minimun loss or high accuracy fastly
# optimisers present are adadelta, adafactor, adagrad, adam, adamw, adamax, ftrl, lion, nadam, optimizer, rmsprof, sgd(sochastic gradient descent)
#loss function is used to find the error value which is predicted - required

_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy*100))
'''
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save("model.h5")
print("Saved model to disk")
'''
data=dataset[760:,0:8]
res = model.predict(data)
print(res)
