# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:17:46 2019

@author: ENIAC
"""
from minisom import MiniSom
import pandas as pd
import numpy as np

#importar a base de dados
base = pd.read_csv('credit-data.csv')

#remover os dados inexistentes (nan)
base = base.dropna()

#substituir as idades inválidas pela média de idades
#para saber  média de idades, digita no Console: base.age.mean()
base.loc[base.age < 0, 'age'] = 40.92

X = base.iloc[:, 0:4].values
y = base.iloc[:, 4].values

#normalizador, para deixar na escala entre 0 e 1
from sklearn.preprocessing import MinMaxScaler
normalizador = MinMaxScaler(feature_range=(0,1))
X = normalizador.fit_transform(X)

#para definir os valores de X e Y
#1 - precisa tirar a raiz quadrada da quantidade total de registros
#2 - multiplica o resultado por 5, que é a qtd de colunas
#o X e o X será o resultado do valor aproximado
#neste exemplo, multiplicando 15 * 15 = 225 que é o valor aproximado do resultado dos cálculos
som = MiniSom(x = 15, y = 15, input_len = 4, random_seed = 0)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

from pylab import pcolor, colorbar, plot
pcolor(som.distance_map().T)
colorbar()

markers = ['o', 's']
colors = ['r', 'g']

#percorrer cada registro
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
         markerfacecolor = 'None', markersize = 10,
         markeredgecolor = colors[y[i]], markeredgewidth = 2)

mapeamento = som.win_map(X)
suspeitos = np.concatenate((mapeamento[(13,9)], mapeamento[(1,10)]), axis = 0)
suspeitos = normalizador.inverse_transform(suspeitos)

classe = []

for i in range(len(base)):
    for j in range(len(suspeitos)):
        if base.iloc[i, 0] == int(round(suspeitos[j, 0])):
            classe.append(base.iloc[i,4])
classe = np.asarray(classe)

suspeitos_final = np.column_stack((suspeitos, classe))
suspeitos_final = suspeitos_final[suspeitos_final[:, 4].argsort()]

