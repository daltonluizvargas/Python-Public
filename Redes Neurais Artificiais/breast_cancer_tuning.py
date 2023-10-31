import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

def criarRede(optimizer, loss, kernel_initializer, activation, neurons):
    classificador = Sequential()

    #units é a quantidade de neurônios da camada oculta - CAMADA DE ENTRADA
    classificador.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer, input_dim = 30))    
    #DROPOUT, evitando o overfitting adicionando uma camada de DROPOUT passando a porcentagem que será zerada
    #vai pegar 20% da camada de entrada e vai zerar estes valores
    classificador.add(Dropout(0.2))
    
    #MAIS UMA CAMADA OCULTA
    classificador.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer))
    classificador.add(Dropout(0.2))

    #MAIS UMA CAMADA OCULTA
    classificador.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer))
    classificador.add(Dropout(0.2))

    #MAIS UMA CAMADA OCULTA
    classificador.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer))
    classificador.add(Dropout(0.2))

    
    #CAMADA DE SAÍDA
    classificador.add(Dense(units = 1,activation = 'sigmoid'))
    
    #configurar a rede neural
    #optimizer é a função que vamos utilizar para fazer o ajuste dos pesos
    #adam é o mais indicado
    #LOSS é o calculo do erro
    #classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    
    classificador.compile(optimizer = optimizer, loss = loss, metrics = ['binary_accuracy'])
    
    return classificador

classificador =KerasClassifier(build_fn=criarRede)
parametros = {'batch_size': [10, 30], 'epochs': [50, 100], 'optimizer': ['adam', 'sgd'], 
              'loss':['binary_crossentropy', 'hinge'], 'kernel_initializer': ['random_uniform', 'normal'], 
              'activation': ['relu', 'tanh'], 'neurons': [16, 8]}

grid_search = GridSearchCV(estimator=classificador, param_grid=parametros, scoring='accuracy', cv = 5)

grid_search = grid_search.fit(previsores, classe)

melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_


