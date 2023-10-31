from minisom import MiniSom
import pandas as pd
import numpy as np

# Carregamento da base de dados
base = pd.read_csv('personagens.csv')
X = base.iloc[:, 0:6].values
y = base.iloc[:,6].values

# Normalização para os dados ficarem entre 0 e 1
from sklearn.preprocessing import MinMaxScaler
normalizador = MinMaxScaler(feature_range = (0,1))
X = normalizador.fit_transform(X)

# Aplicando a fórmula abordada no slide teórico, primeiro tiramos a raíz quadrada
# da quantidade de registros (293), que é igual a 17,11
# Multiplicamos 17,11 por 5 = 85,58
# Vamos definir o mapa auto organizável com as dimensões 9 x 9
# que equivale a 81 neurônios no total
# O input_len possui o valor 6 porque temos 6 entradas
som = MiniSom(x = 9, y = 9, input_len = 6, random_seed = 0,
              learning_rate = 0.5, sigma = 1.0)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 500)

# Transformação das classes em números para associarmos com os markers e colors abaixo
y[y == 'Bart'] = 0
y[y == 'Homer'] = 1

# O código abaixo gera o mapa auto organizável e imprime os símbolos de acordo
# com os valores das classes
from pylab import pcolor, colorbar, plot
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
         markerfacecolor = 'None', markersize = 10,
         markeredgecolor = colors[y[i]], markeredgewidth = 2)
    
# Visualizando o mapa acima, detectamos a coordenada (6,2) como a única com 
# possíveis outliers (cor amarela)
mapeamento = som.win_map(X)
suspeitos = mapeamento[(6,2)]
suspeitos = normalizador.inverse_transform(suspeitos)

# Precisamos agora buscar os possíveis outliers e para isso é necessário
# comparar cada uma das características da base original com a base de suspeitos
# Caso tivéssemos um atributo identificador para cada registro bastaria comparar
# o identificador
classe = []
for i in range(len(base)):
    for j in range(len(suspeitos)):
       if ((base.iloc[i, 0] == suspeitos[j,0]) and
          (base.iloc[i, 1] == suspeitos[j,1]) and
          (base.iloc[i, 2] == suspeitos[j,2]) and
          (base.iloc[i, 3] == suspeitos[j,3]) and
          (base.iloc[i, 4] == suspeitos[j,4]) and
          (base.iloc[i, 5] == suspeitos[j,5])):
           classe.append(base.iloc[i,6])
classe = np.asarray(classe)

# Caso o tamanho da variável suspeitos seja maior que o tamanho da variável
# classe, verifique o arredondamento dos valores por causa da normalização

# Criação da lista final de suspeitos
# Podemos perceber que de fato todos esses registros saem fora do padrão, pois
# são todas imagens do Homer que não apresentam as características do Homer
suspeitos_final = np.column_stack((suspeitos, classe))
suspeitos_final = suspeitos_final[suspeitos_final[:, 4].argsort()]