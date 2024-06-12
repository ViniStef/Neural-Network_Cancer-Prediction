import numpy as np
import pandas as pd 
import tensorflow as tf 
import sklearn as sk 
from tensorflow import keras
from scikeras.wrappers import KerasClassifier 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.model_selection import GridSearchCV    

# 1 - Remoção das Linhas Nulas (NaN/Null)
# 2 - Separação dos dados em X (Variáveis independentes) e y (Variável dependente)
# 3 - Transformação das variáveis categóricas em numéricas (LabelEncoder e OneHotEncoder): (0/1), (0,0,1/0,1,0) 
# 4 - Separação em Teste e Treino
# 5 - Normalização

dataset = pd.read_csv("The_Cancer_data_1500_V2.csv")

dataset =  dataset.dropna()

# print(dataset)

x = dataset.iloc[:, 0:-1]
y = dataset.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)



sc = StandardScaler()

# Faz um média nos valores totais, sendo esta tendo como base o 0 deste modo padronizando-os, ; 

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# DeepLearning: multilayer. (inputLayer/hiddenLayer/outputLayer)

# def create_model(optimizer='adam'):
#     ann = keras.Sequential()
#     ann.add(tf.keras.layers.Dense(units=6, activation='relu', kernel_initializer='he_normal', input_shape=(x_train.shape[1],)))
#     ann.add(tf.keras.layers.Dense(units=6, activation='relu', kernel_initializer='he_normal'))
#     ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
#     ann.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#     return ann

# model = KerasClassifier(build_fn=create_model, verbose=1)

# # Perform grid search
# optimizer = ['SGD', 'Adam']
# batch_size = [16, 32, 64]
# epochs = [10, 20, 30]
# param_grid = dict(optimizer=optimizer, batch_size=batch_size, epochs=epochs)

# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=None, cv=5)
# grid_result = grid.fit(x_train, y_train)

# # Display the best parameters and the corresponding score
# print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

# # Display mean and standard deviation of the scores for each parameter combination
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print(f"{mean:.6f} ({stdev:.6f}) with: {param}")

# def create_model_adam(neurons=6, learn_rate=0.01):
#     ann = keras.Sequential()
#     ann.add(tf.keras.layers.Dense(units=neurons, activation='relu', kernel_initializer='he_normal', input_shape=(x_train.shape[1],)))
#     ann.add(tf.keras.layers.Dense(units=neurons, activation='relu', kernel_initializer='he_normal'))
#     ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
#     optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
#     ann.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#     return ann

# model = KerasClassifier(build_fn=create_model_adam,learn_rate=0.001, neurons=3, epochs=30, batch_size=32, verbose=1)

# learn_rate = [0.001, 0.01, 0.1]
# neurons = [3, 6, 9]
# param_grid = dict(neurons=neurons, learn_rate=learn_rate)

# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=None, cv=5)
# grid_result = grid.fit(x_train, y_train)

# print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

# # Display mean and standard deviation of the scores for each parameter combination
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print(f"{mean:.6f} ({stdev:.6f}) with: {param}")


best_model = keras.Sequential()
best_model.add(tf.keras.layers.Dense(units=9, activation='relu', kernel_initializer='he_normal', input_shape=(x_train.shape[1],)))
best_model.add(tf.keras.layers.Dense(units=9, activation='relu', kernel_initializer='he_normal'))
best_model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
best_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
history = best_model.fit(x_train,y_train, epochs=30, batch_size=16)

loss, accuracy = best_model.evaluate(x_test, y_test)
print("Loss: %.2f" % loss)
print("Accuracy: %.2f" % (accuracy*100))

# Salvar o modelo em JSON
model_json = best_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Salvar os pesos
best_model.save_weights("model.weights.h5")
print("Saved model to disk")

# ann = tf.keras.models.Sequential()

# # Usando camanda Densa, pois cada neuronio aponta para o próximo.

# # Activation Relu: 
# ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# # Adam: 
# ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # batch: Agrupamento de valores para analise
# # epochs: Quantas vezes terá transição entre os neurônios
# ann.fit(x_train, y_train, batch_size=16, epochs=10)

# # Predição dos testes.
# y_pred = ann.predict(x_test)


# # Separa a probalidade de a pessoa ter câncer, esta sendo 60% para verdadeiro.
# y_pred =  (y_pred > 0.6)

# # VN: verdadeiro negativo (Rotulo Zero Correto): Acerta

# # FP:  falso positivo (Rotulo Zero Negativo) : Erra

# # FN: falso negativo (Rotulo Um Negativo) : Erra

# # VP: verdadeiro positivo (Rotolo um Correto) : Acerta

# cm = confusion_matrix(y_test, y_pred)

# print(cm)

# print(dataset['Diagnosis'].value_counts());

