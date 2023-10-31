from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#carregar a base de dados
base = pd.read_csv('ocupacao.csv')

#excluir as linhas com valores em branco ou faltantes (nan)
base = base.dropna()

#definindo a coluna da base de dados que será utilizada para realizar as previsões
base_treinamento = base.iloc[:, 1:2].values

#normalização para colocar os valores numa escala de 0 até 1
normalizador = MinMaxScaler(feature_range=(0,1))

base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

#criando as listas
previsores = []
ocupacao_real = []

#90 registros anteriores, percorrendo a base
for i in range(90, 1989):
    #append adicionar dados a lista, colocar o primeiro registro da base de dados, 
    #ou seja, diminuindo o 90 - 90 vai pegar o registro 0, na segunda vez, 91 - 90, vai pegar o registro 1 e assim por diante
    #0 é o valor da coluna da base_treinamento_normalizada
    previsores.append(base_treinamento_normalizada[i-90:i, 0])    
    ocupacao_real.append(base_treinamento_normalizada[i, 0])
    
#transformar os dados para numpy, para que possa ser utilizado na rede neural
previsores, ocupacao_real = np.array(previsores), np.array(ocupacao_real)

#ESTRUTURA DO RESHAPE keras
#batch_size = quantidade de registros
#timesteps = intervalo de tempo
#input_dim = quantidade de atributos previsores
previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))

regressor = Sequential()

#units equivale ao número de células em memória, deve ser um número grande para adicionar mais
#dimencionalidade em capturar a tendência no decorrer do tempo
#return_sequences = True quando vai ter mais de uma camada LSTM, indica que vai passa a informação pra frente
#para as outras camadas subsequentes
#input_shapes = dados de entrada
# o numero 1 depois do previsores.shape[1] é a quantidade de atributos previsores
regressor.add(LSTM(units = 256, return_sequences = True, input_shape = (previsores.shape[1], 1)))

#Dropout(0.3) significa que vai zerar 30% das entradas para evitar o overfiting
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 128, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 64, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 32))
regressor.add(Dropout(0.3))

regressor.add(Dense(units = 1, activation = 'linear'))

regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',
                  metrics = ['mean_absolute_error'])
regressor.fit(previsores, ocupacao_real, epochs = 100, batch_size = 32)

base_teste = pd.read_csv('ocupacao_teste.csv')
ocupacao_real_teste = base_teste.iloc[:, 1:2].values
base_completa = pd.concat((base['%'], base_teste['%']), axis = 0)
entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values
entradas = entradas.reshape(-1, 1)
entradas = normalizador.transform(entradas)

X_teste = []
for i in range(90, 133):
    X_teste.append(entradas[i-90:i, 0])
X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))
previsoes = regressor.predict(X_teste)
previsoes = normalizador.inverse_transform(previsoes)

previsoes.mean()
ocupacao_real_teste.mean()
    
plt.plot(ocupacao_real_teste, color = 'grey', label = 'Ocupação real')
plt.plot(previsoes, color = 'blue', label = 'Ocupação prevista')
plt.title('Previsão ocupação')
plt.xlabel('Tempo')
plt.ylabel('Valor Sistema Hotel')
plt.legend()
plt.show()


























