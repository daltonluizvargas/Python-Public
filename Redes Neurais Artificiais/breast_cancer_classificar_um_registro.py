import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout


previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

classificador = Sequential()

 #units é a quantidade de neurônios da camada oculta - CAMADA DE ENTRADA
classificador.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'normal', input_dim = 30))    
    #DROPOUT, evitando o overfitting adicionando uma camada de DROPOUT passando a porcentagem que será zerada
    #vai pegar 20% da camada de entrada e vai zerar estes valores
classificador.add(Dropout(0.2))
    
    #MAIS UMA CAMADA OCULTA
classificador.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'normal'))
classificador.add(Dropout(0.2))

    #MAIS UMA CAMADA OCULTA
classificador.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'normal'))
classificador.add(Dropout(0.2))

    #MAIS UMA CAMADA OCULTA
classificador.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'normal'))
classificador.add(Dropout(0.2))

    
    #CAMADA DE SAÍDA
classificador.add(Dense(units = 1,activation = 'sigmoid'))
    
    #configurar a rede neural
    #optimizer é a função que vamos utilizar para fazer o ajuste dos pesos
    #adam é o mais indicado
    #LOSS é o calculo do erro
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

classificador.fit(previsores, classe, batch_size=10, epochs=100)

#NOVAS ENTRADAS
novo = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178, 0.20, 0.05, 1098, 0.87, 4500, 
                  145.2, 0.005, 0.04, 0.05, 0.015, 0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185, 
                  0.84, 158, 0.363]])
previsao = classificador.predict(novo)
previsao = (previsao > 0.9)