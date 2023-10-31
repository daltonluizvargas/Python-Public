import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

def criarRede():
    classificador = Sequential()

    #units é a quantidade de neurônios da camada oculta - CAMADA DE ENTRADA
    classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform', input_dim = 30))    
    #DROPOUT, evitando o overfitting adicionando uma camada de DROPOUT passando a porcentagem que será zerada
    #vai pegar 20% da camada de entrada e vai zerar estes valores
    classificador.add(Dropout(0.2))
    
    #MAIS UMA CAMADA OCULTA
    classificador.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'random_uniform'))
    classificador.add(Dropout(0.2))

    #MAIS UMA CAMADA OCULTA
    classificador.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'random_uniform'))
    classificador.add(Dropout(0.2))

    #MAIS UMA CAMADA OCULTA
    classificador.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'random_uniform'))
    classificador.add(Dropout(0.2))

    
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
    
    return classificador

classificador = KerasClassifier(build_fn=criarRede, epochs = 100, batch_size = 10)

#X são os parâmetros previsores
#Y são as classes
#CV é quantas vezes será feito o teste, ou seja, é o número de folds
#SCORING é como será o retorno do resultado, neste caso será a precisão
resultados = cross_val_score(estimator=classificador, X=previsores, y=classe, cv=10, scoring='accuracy')

media = resultados.mean()

#FAZER O CÁLCULO DO DESVIO PADRÃO, PARA SABER QUANTO OS VALORES ESTÃO VARIANDO EM RELAÇÃO A MÉDIA
#ou seja, quanto os valores estão longe ou perto dessa média
desvio = resultados.std()
