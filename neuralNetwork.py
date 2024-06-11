import numpy as np
import pandas as pd 
import tensorflow as tf 
import sklearn as sk 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

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

ann = tf.keras.models.Sequential()

# Usando camanda Densa, pois cada neuronio aponta para o próximo.

# Activation Relu: 
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Adam: 
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# batch: Agrupamento de valores para analise
# epochs: Quantas vezes terá transição entre os neurônios
ann.fit(x_train, y_train, batch_size=16, epochs=10)

# Predição dos testes.
y_pred = ann.predict(x_test)


# Separa a probalidade de a pessoa ter câncer, esta sendo 60% para verdadeiro.
y_pred =  (y_pred > 0.6)

# VN: verdadeiro negativo (Rotulo Zero Correto): Acerta

# FP:  falso positivo (Rotulo Zero Negativo) : Erra

# FN: falso negativo (Rotulo Um Negativo) : Erra

# VP: verdadeiro positivo (Rotolo um Correto) : Acerta

cm = confusion_matrix(y_test, y_pred)

print(cm)

print(dataset['Diagnosis'].value_counts());

