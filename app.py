import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image 
import traceback

# --- 1. Carregar o Modelo ---
print("Carregando modelo treinado...")
model = tf.keras.models.load_model('modelo_mnist.h5')
print("Modelo carregado com sucesso!")


# --- 2. Função de Classificação (Com Bloco Try/Except) ---
def classificar_digito(imagem_desenhada_dict):
    
    try:
        # VERIFICAÇÃO DE SEGURANÇA
        if imagem_desenhada_dict is None:
            return None # Retorna None para limpar o Label

        imagem_array_bruto = imagem_desenhada_dict['composite']
        
        # 1. Converter o array NumPy (extraído do dict) para uma imagem (PIL)
        # O 'composite' é um array RGBA (4 canais), por isso precisamos do fromarray
        img_pil = Image.fromarray(imagem_array_bruto)
        
        # 2. FORÇAR A CONVERSÃO PARA ESCALA DE CINZA (Modo 'L')
        # Isso descarta os canais de cor e transparência
        img_pil = img_pil.convert('L')
        
        # 3. Redimensionar a imagem para 28x28 (o formato do MNIST)
        img_pil = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
        
        # 4. Converter a imagem de volta para um array NumPy
        imagem_array = np.array(img_pil) 

        # 5. INVERSÃO MANUAL DE CORES
        # O desenho é (preto = 0), o MNIST espera (branco = 255)
        imagem_invertida = 255.0 - imagem_array

        # 6. Normalizar os pixels de [0, 255] para [0.0, 1.0]
        imagem_normalizada = imagem_invertida.astype('float32') / 255.0
        
        # 7. Mudar o formato do array para (1, 28, 28, 1)
        imagem_formatada = np.reshape(imagem_normalizada, (1, 28, 28, 1))

        # 8. Fazer a previsão
        previsao = model.predict(imagem_formatada, verbose=0)
        
        # 9. Formatar a saída para o Gradio
        confiancas = {str(i): float(prob) for i, prob in enumerate(previsao[0])}
        
        return confiancas

    except Exception as e:
        print(f"Erro durante o processamento: {e}")
        traceback.print_exc() # Imprime o erro no terminal para debug
        raise gr.Error(f"Erro no processamento: {e}")


# --- 3. Construir a Interface do Gradio (Usando gr.Blocks) ---
print("Iniciando interface do Gradio com gr.Blocks...")

with gr.Blocks() as app:
    titulo_app = "Reconhecimento de Dígitos MNIST"
    descricao_app = "Desenhe um dígito (0-9) no quadrado, clique em 'Submit' e veja a Rede Neural (CNN) tentar adivinhar qual é!"
    
    gr.Markdown(f"<h1 style='text-align: center;'>{titulo_app}</h1>")
    gr.Markdown(f"<p style='text-align: center;'>{descricao_app}</p>")
    
    with gr.Row():
        interface_entrada = gr.Sketchpad(
            label="Desenhe aqui",
        )

        interface_saida = gr.Label(num_top_classes=3, label="Previsão")

    with gr.Row():
        btn_submit = gr.Button("Submit")
        btn_clear = gr.Button("Clear")

    # --- 4. Definir os Eventos dos Botões ---
    
    btn_submit.click(
        fn=classificar_digito,  # Função a ser chamada
        inputs=interface_entrada, # Pega dados do Sketchpad
        outputs=interface_saida   # Envia dados para o Label
    )
    
    def limpar_tudo():
        return None, None
        
    btn_clear.click(
        fn=limpar_tudo,          # Função a ser chamada
        inputs=None,             # Não precisa de entrada
        outputs=[interface_entrada, interface_saida] # Limpa AMBOS
    )


# Lançar a app
print("Lançando app...")
app.launch(share=True)

