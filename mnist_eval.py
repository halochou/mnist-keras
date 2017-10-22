import keras
from keras.datasets import mnist

with open('model.json') as f:
	model_json_string = f.readline()
	model = keras.models.model_from_json(model_json_string)

model.load_weights('weights.hdf5')
model.summary()
_, (x_test, y_test) = mnist.load_data()

test_img = x_test[0].reshape((1,28,28,1))
predict_result = model.predict(test_img)
print(predict_result, y_test[0])