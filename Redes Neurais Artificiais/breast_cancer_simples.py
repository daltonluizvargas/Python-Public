import pandas as pd

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

import keras

#modelo sequencial
from keras.models import Sequential

#layers são camadas, neste caso usamos camadas densas, que é a ligação de um neurônio com todos os outros
from keras.layers import Dense

classificador = Sequential()

#units é a quantidade de neurônios da camada oculta - CAMADA DE ENTRADA
classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform', input_dim = 30))

#MAIS UMA CAMADA OCULTA
classificador.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'random_uniform'))
#MAIS UMA CAMADA OCULTA
classificador.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'random_uniform'))
#MAIS UMA CAMADA OCULTA
classificador.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'random_uniform'))

#CAMADA DE SAÍDA
classificador.add(Dense(units = 1,activation = 'sigmoid'))

#configurar a rede neural
#optimizer é a função que vamos utilizar para fazer o ajuste dos pesos
#adam é o mais indicado
#LOSS é o calculo do erro
#classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

#MELHORANDO O OTIMIZADOR ADAM
#parâmetros importantes: taxa de aprendizado e função de decaimento do valor da taxa de aprendixagem (DECAY)
otimizador = keras.optimizers.Adam(lr=0.001, decay = 0.0001, clipvalue = 0.5)
classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

#VISUALIZANDO OS PESOS, CONJUNTO DE PESOS QUE A REDE NEURAL CONSEGUE APRENDER
pesos0 = classificador.layers[0].get_weights()
pesos1 = classificador.layers[1].get_weights()
pesos2 = classificador.layers[2].get_weights()
pesos3 = classificador.layers[3].get_weights()
pesos4 = classificador.layers[4].get_weights()

#batch_size será feito a cálculo do erro para 10 registros e depois faz o ajuste dos pesos, 
#depois faz o cáculo para mais 10 registros e faz o ajuste dos pesos e assim por diante
#EPOCHS é quantas vezes que será feito o ajuste dos pesos
classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=100)

#AVALIANDO COM O SKLEARN
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

#AVALIANDO COM O KERAS
resultado = classificador.evaluate(previsores_teste, classe_teste)