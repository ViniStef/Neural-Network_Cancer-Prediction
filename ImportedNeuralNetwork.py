from tensorflow import keras
from keras import models

json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = models.model_from_json(loaded_model_json)

loaded_model.load_weights("model.weights.h5")
print("Loaded model from disk")