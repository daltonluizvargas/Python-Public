from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Aula 1
base = pd.read_csv('BVSP.csv')

#dropna para tirar valores faltantes
base = base.dropna()


base_treinamento = base.iloc[:, 1:2].values

#tranforma para escala de 0 até 1, para visualizar o valor real, tem que desnormalizar
normalizador = MinMaxScaler(feature_range = (0, 1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

# Aula 2
# A cada tempo T a rede neural olhara para os 90 valores anteriores e baseado
# no que ele observar dos dias anteriores ele tentará fazer a previsão
previsores = []
preco_real = []

#começça nos 90 pq pega os 90 anterior - 1242 é o tamanho da base de dados
#i-90 
for i in range(90, 1484):
    previsores.append(base_treinamento_normalizada[i-90:i, 0])
    preco_real.append(base_treinamento_normalizada[i, 0])
previsores, preco_real = np.array(previsores), np.array(preco_real)

# Usado para aumentar as dimensões caso queira usar mais indicadores
# segundo parâmetro é o novo formato, atual tem 2 dimensões e vamos colocar
# em 3 dimensões - mostrar documentação keras
# primeiro parâmetro é a quantidade de registros, o segundo é o número de tempos
# e o terceiro é a quantidade de previsores - intervalo de tempo = timestep
previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))

# Aula 3
regressor = Sequential()
# units número de células de memória, deve ser um número grande para adicionar
# mais dimensionalidade para capturar a tendência no decorrer do tempo
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores.shape[1], 1)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1, activation = 'linear'))

# Mostrar a documentação sobre rmsprop
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', 
                  metrics = ['mean_absolute_error'])

# para esse tipo de problema 100 épocas é uma boa alternativa
#
regressor.fit(previsores, preco_real, epochs = 100, batch_size = 32)

# Aula 4
base_teste = pd.read_csv('BVSP_teste.csv')
preco_real_teste = base_teste.iloc[:, 1:2].values

# o modelo foi treinado usando 90 preços anteriores, por isso precisamos
# os 90 preços anteriores a cada data de janeiro e pra isso precisaremos
# tanto da base de treinamento quanto da base de teste
# com a concatenação vamos pegar os preços das 90 entradas antes de cada
# data de janeiro
# soma a quantidade de registros dos dois datasets
base_completa = pd.concat((base['Open'], base_teste['Open']), axis = 0)

# agora precisamos pegar os 90 registros anteriores a cada data de janeiro
# aqui passamos o índice dentro dos colchetes
# primeiro parâmetro é o lower bound
# o primeiro len retorna a última ação de janeiro, diminuindo do teste
# retorna a última ação de dezembro e menos 90 retorna as últimas que 
# devem ser buscadas - este é o lower bound e o upper colocamos : pra 
# pegar todo o restante da base
entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values
entradas = entradas.reshape(-1,1)

# não chama o fit_transform pra ficar na mesma escala
entradas = normalizador.transform(entradas)

X_teste = []
# 112 é 90 + 22
for i in range(90, 112):
    X_teste.append(entradas[i-90:i, 0])

X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))
previsoes = regressor.predict(X_teste)
previsoes = normalizador.inverse_transform(previsoes)

previsoes.mean()
preco_real_teste.mean()

# Aula 5
plt.plot(preco_real_teste, color = 'red', label = 'Preço real')
plt.plot(previsoes, color = 'blue', label = 'Previsões')
plt.title('Previsão preço das ações')
plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legend()
plt.show()
