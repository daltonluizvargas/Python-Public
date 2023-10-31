from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np

#1ª camada de convolução
#Parametro 32 = número de filtros, recomendavel é 64 filtros
#parametro 3,3 = dimensõe do detector
#parametro inputShape = altura, largura, canais
#parametro relu para tirar os valores negativos,
#ou seja, os valores que representam as partes mais escuras da imagem
#parametro batchNormalization para acelerar o processamento, deixando numa escala entre 0 e 1
#parametro pool_size é uma matriz de 4 pixels utilizada para pegar o maior valor
classificador = Sequential()
classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size= (2,2)))

#2ª camada
classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size= (2,2)))

#flatten transforma matriz em um vetor para passar como entrada para a rede neural
classificador.add(Flatten())
#rede neural convulocional está pronta até aqui

#CAMADAS OCULTAS
#criar a rede neural densa
#units é a quantidade de neuronios nessa primeira camada oculta
#dropOut para zerar 20% das entradas
classificador.add(Dense(units=128, activation= 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=128, activation= 'relu'))
classificador.add(Dropout(0.2))
#camada de saída
#somente uma saída por ser um problema de classificação binária
classificador.add(Dense(units=1, activation= 'sigmoid'))

classificador.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])

#normalização dos pixels
gerador_treinamento = ImageDataGenerator(rescale=1./255, rotation_range=7, horizontal_flip=True,
                                         shear_range=0.2, height_shift_range=0.07, zoom_range=0.2)

gerador_teste = ImageDataGenerator(rescale=1./255)

base_treinamento = gerador_treinamento.flow_from_directory('dataset/training_set', target_size=(64,64), 
                                                        batch_size=32, class_mode='binary')

base_teste = gerador_teste.flow_from_directory('dataset/test_set', target_size=(64,64), 
                                              batch_size=32, class_mode='binary')

classificador.fit_generator(base_treinamento, steps_per_epoch=4000/32, epochs=5, 
                            validation_data=base_teste, validation_steps=1000/32)

imagem_teste = image.load_img('dataset/test_set/gato/cat.3500.jpg', target_size=(64,64))
imagem_teste = image.img_to_array(imagem_teste)
imagem_teste /= 255
imagem_teste = np.expand_dims(imagem_teste, axis = 0)
previsao = classificador.predict(imagem_teste)
previsao = (previsao > 0.3)
base_treinamento.class_indices
