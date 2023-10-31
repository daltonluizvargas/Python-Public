import pandas as pd
import keras
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

classificador_json = classificador.to_json()
with open('classificador_breast.json', 'w') as json_file:
    json_file.write(classificador_json)

#SALVAR OS PESOS
    #caso ocorra algum erro, o pacote H5 deve ser instalado com o anaconda prompt
    #pip install h5py
classificador.save_weights('classificador_breast.h5')