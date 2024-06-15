import pandas as pd 
import tensorflow as tf  
from tensorflow import keras
from scikeras.wrappers import KerasClassifier 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV    

"""
Processos a serem seguidos: ----- Perceptron Multicamadas (Multilayer Perceptron (MLP)) -----

# 1 - Remoção das Linhas Nulas (NaN/Null)
# 2 - Separação dos dados em X (Variáveis independentes) e y (Variável dependente)
# 3 - Transformação das variáveis categóricas em numéricas (LabelEncoder e OneHotEncoder): (0/1), (0,0,1/0,1,0) 
# 4 - Separação em Teste e Treino
# 5 - Normalização

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

"""
Definir os conjuntos de dados, reservando para 'x' 80% dos dados e 20% para 'y'.

"""
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


"""
Instancia o StandardScaler, que é responsável por padronizar os dados. Ele ajusta as variáveis independentes para que todas tenham o mesmo peso durante o treinamento, fazendo com que tenham média zero e desvio padrão aproximadamente igual a 1, garantindo que tenham a mesma escala.
"""
sc = StandardScaler()

# Faz um média nos valores totais, sendo esta tendo como base o 0 deste modo padronizando-os. 
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

"""
Função para criar o modelo inicial que será utilizado para testar os hiperparâmetros, inicializado com valores padrão.
"""
def create_model(neurons=6, learn_rate=0.01, optimizer='adam'):
    """
    Adicionamos 2 camadas internas (Hidden layers), com neurônios que serão passados pelo GridSearchCV, utilizamos a função de ativação ReLU porque é fácil de calcular, ajuda a previnir overfitting e auxilia a manter os valores do gradiente estáveis durante o treinamento da rede neural, podendo ser definida por: ReLU(x)=max(0,x).
    Utilizamos kernel_initializer='he_normal' para inicializar os pesos da rede neural. O método He normal é adequado para redes que utilizam a função de ativação ReLU, pois ajuda a manter a estabilidade do treinamento.
    """
    inputs = tf.keras.layers.Input(shape=(x_train.shape[1],))
    x = tf.keras.layers.Dense(neurons, activation='relu', kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.Dense(neurons, activation='relu', kernel_initializer='he_normal')(x)
    """
    Por ser uma rede de classificação binária, utilizamos apenas um neurônio de saída em conjunto com a função sigmoide. A função sigmoide é ideal para esse tipo de problema, pois a saída será um valor entre 0 e 1, indicando a probabilidade de um elemento pertencer a uma determinada classe. Com base em um limiar simples (por exemplo, Ter câncer > 0,6), podemos classificar o elemento como pertencente ou não àquela classe. 
    """
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    """
    Verifica qual otimizador está sendo testado no momento atual para indicar ao Keras qual será utilizado na função de criação. Estes otimizadores são responsáveis pela formula que será utilizada na atualização dos pesos durante o treinamento da rede neural, Sendo Adam e SGD os mais eficientes principalmente para redes de classificação binária.
    """
    if optimizer.lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
    elif optimizer.lower() == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learn_rate)
    else:
        raise ValueError("Optimizer not supported")
    

    """
    Compilamos os parâmetros, indicando a perda(loss) como 'binary_crossentropy', já que é uma decisão binária, e, indicamos em 'metrics' qual o foco desse modelo, sendo 'accuracy' um indicador que queremos o resultado focado em precisão.
    """
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

"""
Colocamos o modelo criado acima dentro de um KerasClassifier, que nos permite utilizá-lo como um estimador do scikit-learn, que irá ser utilizado no GridSearchCV como parâmetro principal na definição da rede que teremos.
"""
model = KerasClassifier(model=create_model,learn_rate=0.001,neurons=3, verbose=1)

"""
Definimos os parâmetros que serão utilizados no GridSearchCV.
"""
param_grid = {
    # Quantidade de neurônios em cada camada interna(Hidden Layer).
    'neurons': [3, 6, 9],
    # Controla a rapidez e eficiência com que o modelo aprende a partir dos dados.
    'learn_rate': [0.001, 0.01, 0.1],
    # Fórmula que será utilizada para atualizar os pesos.
    'optimizer': ['SGD', 'Adam'],
    # Quantidade de dados que serão analisados de uma vez e utilizados para ajustar os pesos do modelo durante o treinamento.
    'batch_size': [16, 32, 64],
    # Quantidade de vezes que o conjunto de dados terá seus pesos variados. Uma época(epoch) é concluida quando todo o conjunto de dados foi visto.
    'epochs': [10, 20, 30]
}

"""
Performamos a busca em grid, que será o responsável por utilizar o modelo que criamos previamente e testar as diferentes combinações de parâmetros.Isso nos possibilita identificar as melhores combinações que podem ser utilizadas para maximizar a precisão e minimizar a perda (loss).
O parâmetro 'n_jobs=None' indica que o cálculo será executado na CPU principal, utilizando apenas um núcleo. Isso acontece porque 'None' é um indicativo para usar o padrão do GridSearchCV, que geralmente é '1'.
O parâmetro cv=5 indica quantas dobras serão utilizadas na validação cruzada (Cross-validation). A validação cruzada envolve dividir o conjunto de dados em partes, alternando entre treino e teste em cada iteração. Com cv=5, o processo é realizado da seguinte maneira: o conjunto de dados é dividido aleatoriamente em 5 partes iguais. Em cada iteração da validação cruzada, 4 partes são usadas para treinar o modelo e 1 parte é reservada para avaliação do modelo (conjunto de teste). Esse processo é repetido 5 vezes, garantindo que cada parte seja usada uma vez como conjunto de teste. Isso ajuda o modelo a se adaptar a diferentes padrões nos dados e a evitar o overfitting, ao proporcionar uma avaliação mais robusta do desempenho do modelo em dados não vistos.
"""
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=None, cv=5)

"""
Iniciar o processo de busca em grid, otimizando os hiperparâmetros do modelo com base nos dados de treino.
"""
grid_result = grid.fit(x_train, y_train)

"""
Nos mostra quais os melhores parâmetros com seus valores correspondentes.
"""
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

best_batch_size = grid_result.best_params_['batch_size']
best_epochs = grid_result.best_params_['epochs']
best_learn_rate = grid_result.best_params_['learn_rate']
best_neurons = grid_result.best_params_['neurons']
best_optimizer = grid_result.best_params_['optimizer']

"""
Utilizamos um for loop para exibir as informações de cada iteração sobre as combinações que foram testadas durante a busca em grid. Essa abordagem nos permite entender combinações que podem ser
prejudiciais ao modelo atual e também possíveis aspectos que podem afetar os resultados.
"""
# Media do valor da validação cruzada para essa combinação de parâmetros
means = grid_result.cv_results_['mean_test_score']
# Desvio padrão do valor da validação cruzada
stds = grid_result.cv_results_['std_test_score']
# Parâmetros que foram testados
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f"{mean:.6f} ({stdev:.6f}) with: {param}")
    

"""
Função para criar o melhor modelo final, sendo este inicializado com valores padrão.
"""
def create_best_model(learning_rate=0.01, neurons=6, epochs=10, batch_size=16, optimizer='adam'):
    """
    Segue o mesmo padrão de criação da função de modelo teste.
    """
    inputs = tf.keras.layers.Input(shape=(x_train.shape[1],))
    x = tf.keras.layers.Dense(neurons, activation='relu', kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.Dense(neurons, activation='relu', kernel_initializer='he_normal')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    final_model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    if optimizer.lower() == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer.lower() == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError("Otimizador não suportado. Por favor, escolha entre 'adam' ou 'sgd'.")
    
    """
    Informações relacionadas à construção deste modelo.
    """
    model_settings = f"Construído usando: Learning rate: {learning_rate} | Neurons: {neurons} | Epochs: {epochs} | Batch Size: {batch_size} | Optimizer: {optimizer}"
    
    """
    Compila o modelo final com as informações passadas.
    """
    final_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    """
    Guarda um histórico sobre as métricas e perdas durante as épocas do aprendizado.
    """
    history = final_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return final_model, model_settings, history


trained_model, model_settings, history  = create_best_model(learning_rate=best_learn_rate, neurons=best_neurons, epochs=best_epochs, batch_size=best_batch_size, optimizer=best_optimizer)

"""
Calcula a perda e a precisão do modelo baseado nos dados de teste que foram armazenados previamente.
"""
loss, accuracy = trained_model.evaluate(x_test, y_test)
print(model_settings)
print("Loss: %.2f" % loss)
print("Accuracy: %.2f" % (accuracy*100))

"""
Começa o processo de predição, baseado nas variáveis independentes definidas.
"""
y_pred = trained_model.predict(x_test)

"""
Configura a probabilidade da pessoa ter câncer, esta sendo 60% para verdadeiro.
"""
y_pred =  (y_pred > 0.6)

"""
Cria uma matriz de confusão para entender os resultados esperados e os resultados dados pela rede em todos os casos de teste.
VN: Verdadeiro Negativo (Rotulo Zero Correto): Acerta

FP:  Falso Positivo (Rotulo Zero Negativo) : Erra

FN: Falso Negativo (Rotulo Um Negativo) : Erra

VP: Verdadeiro Positivo (Rotulo um Correto) : Acerta
"""
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Total de pacientes com e sem cancêr.
print(dataset['Diagnosis'].value_counts());

"""
Configurações do matplot para criar os gráficos.
"""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

"""
Matriz de confusão das classificações preditas e das classificações verdadeiras.
"""
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16}, ax=ax1)
ax1.set_xlabel('Predicted Labels')
ax1.set_ylabel('True Labels')
ax1.set_title('Confusion Matrix')

"""
Histórico de perda e precisão combinados em função das épocas percorridas.
"""
loss_history = history.history['loss']
accuracy_history = history.history['accuracy']

ax2.plot(loss_history, label='Loss')
ax2.plot(accuracy_history, label='Accuracy', linestyle='--')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Métricas de Desempenho')
ax2.set_title('Histórico de Perda e Precisão durante o Treinamento')

"""
Legenda para o histórico de perda e precisão.
""" 
ax2.legend()

"""
Ajusta o layout para evitar sobreposição.
"""
plt.tight_layout()

"""
Mostra os plots.
"""
plt.show()

"""
Salvar o modelo já treinado em formato JSON
"""
trained_model_json = trained_model.to_json()
with open("trained_model.json", "w") as json_file:
    json_file.write(trained_model_json)

"""
Salvar os pesos associados com o modelo
"""
trained_model.save_weights("trained_model.weights.h5")





















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




# # Salvar o modelo em JSON
# model_json = best_model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)

# # Salvar os pesos
# best_model.save_weights("model.weights.h5")
# print("Saved model to disk")

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

