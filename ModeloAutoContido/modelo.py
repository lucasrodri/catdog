# Suppress all autograph warnings from Tensorflow
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR) # this goes *before* tf import
import tensorflow as tf

from tensorflow import keras
from PIL import Image
import numpy as np
import json
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Exibir apenas mensagens de erro do TensorFlow

# Criar um --help
if len(sys.argv) == 2 and sys.argv[1] == "--help":
    print("Usage: python modelo.py <caminho_para_modelo.h5>")
    sys.exit(0)

# Verificar se o número correto de argumentos foi fornecido
if len(sys.argv) != 2:
    print("Usage: python modelo.py <caminho_para_modelo.h5>")
    sys.exit(1)

# Carregar o modelo
modelo_path = sys.argv[1]
model = keras.models.load_model(modelo_path)

# Entrar em um loop para receber entradas do usuário
while True:
    # Obter entrada do usuário
    #"Digite um JSON de entrada (ou Ctrl+C para sair): "
    entrada_json = input()
    if entrada_json == "quit":
        break

    try:
        # Analisar o JSON de entrada
        entrada = json.loads(entrada_json)
        
        # Verificar se o serviço e o caminho da imagem estão presentes
        if "service" in entrada and "img_path" in entrada:
            # Carregar a imagem
            image_path = entrada["img_path"]
            try:
                image = Image.open(image_path)
                image = image.resize((180, 180))

                # Converter a imagem para um array numpy
                image_np = np.array(image)
                image_np = np.expand_dims(image_np, axis=0)  # Adicionar uma dimensão extra para representar o batch

                # Realizar a previsão com o modelo criado/carregado
                predicao = model.predict(image_np)
                objeto_predita = 1 if predicao[0][0] < 0.5 else 0
                classe_predita = "cat" if predicao[0][0] < 0.5 else "dog"

                # Gerar saída
                saida = {
                    "object_id": objeto_predita,
                    "object_name": classe_predita
                }

                # Imprimir saída
                print(json.dumps(saida))
            except FileNotFoundError:
                # Imprimir mensagem de erro em json
                print(json.dumps({"error": "image not found"}))
        else:
            # Imprimir mensagem de erro em json
            print(json.dumps({"error": "invalid JSON input, make sure to include 'img_path'"}))
            
    
    except json.JSONDecodeError as e:
        # Imprimir mensagem de erro em json
        print(json.dumps({"error": "invalid JSON input", "details": str(e)}))
