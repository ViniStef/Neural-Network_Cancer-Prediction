import numpy as np
import pandas as pd 
import tensorflow as tf 
import sklearn as sk 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)



sc = StandardScaler()

# Faz um média nos valores totais, sendo esta tendo como base o 0 deste modo padronizando-os, ; 

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# DeepLearning: multilayer. (inputLayer/hiddenLayer/outputLayer)

print(x_train)