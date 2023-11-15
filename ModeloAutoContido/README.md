# Modelo Auto Contido

Uma vez com o modelo treinado, podemos exportá-lo para um arquivo que pode ser carregado e usado para fazer previsões.

Utilizaremos o pyinstaller para criar um executável que pode ser usado para fazer previsões.

```bash
pip install pyinstaller
pyinstaller --onefile modelo.py
```

Será criado um arquivo chamado modelo dentro da pasta `dist`. Esse arquivo pode ser copiado para qualquer lugar e usado para fazer previsões. Adicionalmente, temos que copiar o arquivo `h5` que contém o modelo treinado para o mesmo diretório e executar o arquivo modelo da seguinte forma:

```bash
./modelo <modelo.h5>
```

Em seguida, podemos fazer previsões usando o modelo. O modelo espera entradas no stdin em json. O formato esperado é o seguinte:

```json
{"service" : "DogCat","img_path": "dog.jpg"}
```

A saída do modelo é um json com o seguinte formato:

```json
{"object_id": 0, "object_name": "dog"}
```

Caso a imagem não seja encontrada, o modelo retorna o seguinte json de erro:

```json
{"error": "image not found"}
```

Caso o json não seja válido, o modelo retorna o seguinte json de erro:

```json
{"error": "invalid json"}
```
ou 
```json
{"error": "invalid JSON input, make sure to include 'img_path'"}
```

# Exemplo de uso

```bash
./modelo catdog.h5
{"service" : "DogCat","img_path": "dog.jpg"}
{"object_id": 0, "object_name": "dog"}
{"service" : "DogCat","img_path": "cat.jpg"}
{"object_id": 1, "object_name": "cat"}
{"service" : "DogCat","img_path": "eu.jpg"}
{"object_id": 0, "object_name": "dog"}
{"service" : "DogCat","img_path": "teste.jpg"}
{"error": "image not found"}
{"service" : "DogCat","img_path": "dog
{"error": "invalid JSON input", "details": "Unterminated string starting at: line 1 column 35 (char 34)"}
blabla
{"error": "invalid JSON input", "details": "Expecting value: line 1 column 1 (char 0)"}
{"service" : "DogCat","img_path": "dog.jpg"}
{"object_id": 0, "object_name": "dog"}
quit
```