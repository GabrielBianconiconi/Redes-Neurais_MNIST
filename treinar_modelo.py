# 1. Imports necessários
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

print("TensorFlow importado. Versão:", tf.__version__)

# 2. Carregar e Pré-processar os Dados (MNIST)
print("Carregando dataset MNIST...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Pré-processamento das imagens (X)
# Nossas imagens são 28x28 pixels em escala de cinza.
# A CNN espera um 4º eixo para o "canal de cor".
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Normalizar os pixels de [0, 255] para [0.0, 1.0]
# Redes neurais funcionam melhor com valores pequenos.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print(f"Dados de treino: {x_train.shape[0]} amostras")
print(f"Dados de teste: {x_test.shape[0]} amostras")

# Pré-processamento dos rótulos (Y)
# Nossos rótulos são números de 0 a 9.
# A rede neural precisa do formato "one-hot encoding".
# Ex: O número 5 vira [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# 3. Construir a Arquitetura da Rede Neural (CNN)
print("Construindo o modelo da CNN...")
model = Sequential()

# Camada de Convolução: "Olha" para a imagem e extrai features (bordas, curvas).
# 32 filtros, kernel 3x3, função de ativação 'relu'.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# Camada de Pooling: Reduz o tamanho da imagem, mantendo as features importantes.
model.add(MaxPooling2D((2, 2)))

# Mais uma camada de convolução para aprender features mais complexas.
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# "Achatar" os dados 2D para um vetor 1D para conectar na rede neural densa.
model.add(Flatten())

# Camada Densa (Totalmente Conectada): Onde a "mágica" da classificação acontece.
model.add(Dense(128, activation='relu'))
# Dropout: Técnica para evitar que o modelo "decore" os dados (overfitting).
model.add(Dropout(0.5))

# Camada de Saída: 10 neurônios (um para cada dígito, 0-9).
# 'softmax' garante que as saídas somem 1 (interpreta como probabilidades).
model.add(Dense(10, activation='softmax'))

# 4. Compilar o Modelo
# 'adam' é um otimizador eficiente.
# 'categorical_crossentropy' é a função de perda correta para classificação com one-hot.
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Mostra um resumo da arquitetura no terminal
model.summary()

# 5. Treinar o Modelo
print("\nIniciando o treinamento...")
# batch_size: Quantas imagens o modelo vê antes de atualizar os pesos.
# epochs: Quantas vezes o modelo verá o dataset inteiro.
# validation_data: Dados de teste para checar a performance em tempo real.
history = model.fit(x_train, y_train_cat, 
                    epochs=10, 
                    batch_size=128, 
                    validation_data=(x_test, y_test_cat))

print("Treinamento concluído!")

# 6. Salvar o Modelo Treinado
model.save('modelo_mnist.h5')
print("Modelo salvo em 'modelo_mnist.h5'")

# 7.Gerar gráficos para a apresentação
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Acurácia de Treino')
plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
plt.title('Acurácia do Modelo')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Perda de Treino')
plt.plot(history.history['val_loss'], label='Perda de Validação')
plt.title('Perda do Modelo')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()

plt.savefig('graficos_treinamento.png')
print("Gráficos do treinamento salvos em 'graficos_treinamento.png'")
# plt.show()