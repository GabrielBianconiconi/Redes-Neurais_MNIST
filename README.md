# 🤖 Reconhecimento de Dígitos MNIST c

Este projeto utiliza uma **Rede Neural**, construída com TensorFlow/Keras, para reconhecer dígitos manuscritos (0-9).

O projeto é dividido em duas partes principais:
1.  **`treinar_modelo.py`**: Um script que constrói, treina e salva o modelo de CNN usando o famoso dataset MNIST.
2.  **`app.py`**: Uma aplicação web interativa, construída com Gradio, que carrega o modelo treinado e permite que o usuário desenhe um dígito para obter uma previsão em tempo real.

### 🚀 Link da Apresentação (Canva)

Para mais detalhes sobre o projeto, acesse nossa apresentação no Canva:

[Apresentação do Projeto - Reconhecimento de Dígitos (Canva)](https://www.canva.com/design/DAG3wv1yZcQ/_ud0nIhJC1pNM-TG4jjDdQ/edit?utm_content=DAG3wv1yZcQ&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

## 📊 Resultados do Treinamento

O script de treinamento (`treinar_modelo.py`) gera os seguintes gráficos de performance, demonstrando um modelo com alta acurácia e sem *overfitting*:

![Gráficos de Acurácia e Perda do Treinamento]([httpsa://i.imgur.com/g880Fq3.png](https://github.com/GabrielBianconiconi/Redes-Neurais_MNIST/blob/main/graficos_treinamento.png))


## 📂 Estrutura do Projeto

```
/reconhecimento-digitos-mnist
│
├── 📜 treinar_modelo.py   # Script para treinar a CNN e salvar o .h5
├── 🚀 app.py              # Script para rodar a interface web com Gradio
├── 🧠 modelo_mnist.h5     # (Gerado) O modelo treinado
├── 📊 graficos_treinamento.png # (Gerado) Gráficos de performance
└── 📄 README.md           # Este arquivo
```

## 🛠️ Requisitos

Para rodar este projeto, você precisará das seguintes bibliotecas Python. É altamente recomendado usar um ambiente virtual (`venv`).

* `tensorflow`
* `gradio`
* `numpy`
* `matplotlib`
* `pillow` (para processamento de imagem no `app.py`)

Você pode instalar todas as dependências com o pip:

```bash
pip install -r requirements.txt       
```

## 🚀 Como Usar

O fluxo de trabalho é simples: primeiro, treine o modelo; em seguida, execute a aplicação web.

### Passo 1: Treinar o Modelo

Execute o script `treinar_modelo.py` no seu terminal. Este script irá:
1.  Baixar o dataset MNIST.
2.  Construir a arquitetura da CNN.
3.  Treinar o modelo por 10 épocas.
4.  Salvar o modelo treinado como `modelo_mnist.h5`.
5.  Salvar os gráficos de performance como `graficos_treinamento.png`.

```bash
python treinar_modelo.py
```

### Passo 2: Executar a Aplicação Web

Após o arquivo `modelo_mnist.h5` ser criado, execute o script `app.py`.

```bash
python app.py
```

### Passo 3: Testar no Navegador

O script `app.py` irá iniciar um servidor local e fornecer um link (normalmente `http://127.0.0.1:7860`). Abra este link no seu navegador:

1.  Desenhe um dígito (de 0 a 9) na caixa "Desenhe aqui".
2.  Clique no botão **"Submit"**.
3.  O modelo fará a previsão e mostrará os resultados (com as 3 maiores confianças) na caixa "Previsão".
4.  Use o botão **"Clear"** para limpar o desenho e a previsão.

5.  ## 🧠 Arquitetura do Modelo (CNN)

Nosso modelo é uma **Rede Neural Convolucional** (`Sequential`) construída com Keras. A arquitetura é empilhada na seguinte ordem para processar as imagens 28x28:

1.  **`Conv2D`**: Camada "visual" inicial.
    * **Filtros:** 32
    * **Função:** Detectar características de baixo nível (bordas, curvas).
2.  **`MaxPooling2D`**:
    * **Função:** Reduzir o tamanho da imagem ("encolher"), mantendo apenas as características mais fortes.
3.  **`Conv2D`**: Segunda camada "visual".
    * **Filtros:** 64
    * **Função:** Usar as características simples para detectar padrões mais complexos (círculos, linhas completas).
4.  **`MaxPooling2D`**:
    * **Função:** Reduzir o tamanho novamente.
5.  **`Flatten`**:
    * **Função:** "Achatar" o mapa 2D de características em um vetor 1D (uma "lista") para alimentar o "cérebro" da rede.
6.  **`Dense`**: A principal camada "pensante".
    * **Neurônios:** 128
    * **Função:** Analisar a combinação de todos os padrões encontrados para tomar uma decisão.
7.  **`Dropout`**:
    * **Taxa:** 0.5 (50%)
    * **Função:** Técnica de regularização para prevenir *overfitting* (evitar que o modelo "decore" os dados).
8.  **`Dense` (Camada de Saída)**:
    * **Neurônios:** 10
    * **Função:** Classificar a imagem em um dos 10 dígitos (0-9) usando `softmax` para gerar probabilidades.

**Total de Parâmetros Treináveis:** 225.034

## 📊 Dataset: MNIST (Treino e Teste)

Para treinar nossa rede, utilizamos o famoso dataset **MNIST**.

* **Tamanho das Imagens:** Todas as imagens são em escala de cinza e padronizadas no tamanho de **28x28 pixels**.
* **Total de Amostras:** O dataset completo contém **70.000 imagens** no total.
* **Divisão dos Dados:** O Keras já nos entrega o dataset pré-dividido em dois conjuntos distintos que não se sobrepõem:
    * **Conjunto de Treinamento (`x_train`): 60.000 imagens (~85.7%)**
        * **Uso:** Material que o modelo "estuda" durante o `model.fit()`.
    * **Conjunto de Teste (`x_test`): 10.000 imagens (~14.3%)**
        * **Uso:** Passado para o `validation_data`. O modelo *nunca* aprende com essas imagens; elas são usadas apenas como uma "prova final" ao fim de cada época para garantir que o modelo está generalizando e não apenas decorando.

### 👨‍💻 Integrantes do Grupo

* **Gabriel Bianconi** (RA: 20.00822-8)
* **Carlos Alberto Matias da Costa** (RA: 20.01308-6)
* **Bruno Fevereiro** (RA: 20.02194-0)
