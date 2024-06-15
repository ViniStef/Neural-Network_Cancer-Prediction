from keras import models
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

"""
Abre o modelo previamente treinado
"""
json_file = open("trained_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()

"""
Carrega o modelo para variável 'trained_model', permitindo fazer predições com novos dados sem precisar treinar e avaliar o modelo novamente.
"""
trained_model = models.model_from_json(loaded_model_json)

"""
Carrega para o modelo as variáveis previamente definidas.
"""
trained_model.load_weights("trained_model.weights.h5")

"""
O modelo pode ser chamado aqui e usado normalmente.
"""

"""
Acessa o arquivo onde estão os dados.
"""
dataset = pd.read_csv("The_Cancer_data_1500_V2.csv")

"""
Remove qualquer linha que tenha dados faltando ou dados como NaN(Not A Number).
"""
dataset =  dataset.dropna()

"""
x (Variáveis independentes) -> Todas as linhas, todas a colunas menos a última.
"""
x = dataset.iloc[:, 0:-1]
"""
y (Variável dependente) -> Todas as linhas, apenas a última coluna(Diagnosis).
"""
y = dataset.iloc[:, -1]

sc = StandardScaler()
x_scaled = sc.fit_transform(x)

y_pred = trained_model.predict(x_scaled)
y_pred =  (y_pred > 0.6)

cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16})
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()