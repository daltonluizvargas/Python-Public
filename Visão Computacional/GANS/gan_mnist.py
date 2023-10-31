import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Flatten, Reshape
from keras.regularizers import L1L2

#fazer a instalação do keras versão: 2.1.2
#pip install keras==2.1.2
#para instalar o keras_adversarial
#pelo anacondaPrompt: pip install keras_adversarial --user
from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling

self._feed_loss_fns = self.loss_functions
self._feed_inputs = self.inputs
self._feed_input_names = self.input_names
self._feed_input_shapes = self.internal_input_shapes
self._feed_outputs = self.outputs
self._feed_output_names = self.output_names
self._feed_output_shapes = self.internal_output_shapes
self._feed_sample_weights = self.sample_weights
self._feed_sample_weight_modes = self.sample_weight_modes

#carregar a base de dados
(previsores_treinamento, _), (_, _) = mnist.load_data()

#normalização dos dados, tranformando em uma escala entre 0 e 1
previsores_treinamento = previsores_treinamento.astype('float32') / 255

#/////////////////////////////////////////////////////////////////////////////
#rede neural - GERADOR
gerador = Sequential()

#Camadas ocultas
#Regularizador é usado para evitar o overfithing
#1e-5 é um valor muito pequeno, é recomendado na documentação desta biblioteca
gerador.add(Dense(units = 500, input_dim = 100, activation = 'relu', 
                  kernel_regularizer = L1L2(1e-5, 1e-5)))
gerador.add(Dense(units = 500, input_dim = 100, activation = 'relu', 
                  kernel_regularizer = L1L2(1e-5, 1e-5)))

#Camada de Saída
#como o tamanho das imagens é de 28x28, multiplica este valor e o resultado será o valor da saída
#ou seja, 28x28 = 784 pixels
#Função Sigmoid para retornar valores entre 0 e 1, pois este é um problema de classificação binária
gerador.add(Dense(units = 784, activation = 'sigmoid', kernel_regularizer = L1L2(1e-5, 1e-5)))

#Dimensionamento, retorno da função sigmoid e transformar em imagens
gerador.add(Reshape((28,28)))

#/////////////////////////////////////////////////////////////////////////////
# rede neural - DISCRIMINADOR
discriminador = Sequential()

#vai receber imagens 28,28
discriminador.add(InputLayer(input_shape=(28,28)))

#pega a matriz e tranforma novamente em vetor
discriminador.add(Flatten())

#camadas
discriminador.add(Dense(units = 500, activation = 'relu', kernel_regularizer = L1L2(1e-5, 1e-5)))

#mais uma camada
discriminador.add(Dense(units = 500, activation = 'relu', kernel_regularizer = L1L2(1e-5, 1e-5)))

#camada de saída, Units é igual a um, pq teremos somente uma saída
discriminador.add(Dense(units = 1, activation = 'sigmoid', kernel_regularizer = L1L2(1e-5, 1e-5)))

# parametro normal_latent_sampling gerar 100 imagens de dígitos e de não dígitos
gan = simple_gan(gerador, discriminador, normal_latent_sampling((100,)))

#player_params são os pesos de cada uma dessas redes neurais
modelo = AdversarialModel(base_model = gan,
                          player_params = [gerador.trainable_weights, 
                                           discriminador.trainable_weights])
#complilação e o otimizador onde vai atualizar cada cada uma das redes neurais simultaneamente em cada um dos batchs(batchs é a quantidade de registros usados para fazer a atualização dos pesos)
modelo.adversarial_compile(adversarial_optimizer = AdversarialOptimizerSimultaneous(),
                           player_optimizers = ['adam', 'adam'],
                           loss = 'binary_crossentropy')
#TRAINAMENTO
#quanto maior o número de épocas é melhor
modelo.fit(x = previsores_treinamento, y = gan_targets(60000), epochs = 100, batch_size = 256)

amostras = np.random.normal(size = (20,100))
previsao = gerador.predict(amostras)
for i in range(previsao.shape[0]):
    plt.imshow(previsao[i, :], cmap='gray')
    plt.show()
