from numpy import loadtxt
from keras.models import model_from_json

dataset = loadtxt("pima-indians-diabetes.csv",delimiter=',')

data = dataset[758:,0:8]

json_file = open("model.json","r")
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
model.load_weights("model.h5")

prediction = model.predict(data)

for i in dataset[758:,8]:
    print(i,end=' ')
print("")
for i in prediction:
    if i >0.5:
        print(1.0,end=' ')
    else:
        print(0.0,end=' ')
    