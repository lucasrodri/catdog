# Importando Dados

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

from PIL import Image
import numpy as np
import glob, random, ipyplot

# download do dataset
#!curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip

# Remover imagens corrompidas
import os
num_skipped = 0
for folder_name in ("Cat", "Dog"):
  folder_path = os.path.join("PetImages", folder_name)
  for fname in os.listdir(folder_path):
    fpath = os.path.join(folder_path, fname)
    try :
      fobj = open(fpath, "rb")
      is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
    finally:
      fobj.close()

    if not is_jfif:
      num_skipped += 1
      #Delete corrupted images
      os.remove(fpath)
print("Deleted %d images" % num_skipped)

# Criando as datasets
# Cria datasets
image_size = (180, 180)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

# Realizando o Data augmentation
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)
train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

import matplotlib.pyplot as plt
plt.style.use('ggplot')

#função para visualizar o historico de treinamento da rede
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)
    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    
    #Adicione camadas aqui
    #conv
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    #maxpool
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    
    #conv
    x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    #maxpool
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    #flatten
    x = layers.Flatten()(x)
    #Hidden Layers
    #x = layers.Dense(128, activation='sigmoid')(x)
    #x = layers.Dense(32, activation='relu')(x)

    # seleciona a ativação do output baseado no numero de classes
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

print("Criando o modelo...")

model = make_model(input_shape=image_size + (3,), num_classes=2)
model.summary()
keras.utils.plot_model(model, show_shapes=True)


epochs = 20

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


print("Treinando o modelo...")

history_1 = model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds, verbose=1,
)
plot_history(history_1)


"""
print("Carregando os pesos para o modelo feito...")
chekpoint = "save_at_18.h5"

# Loads the weights
model.load_weights(chekpoint)
"""


print("Salvando o modelo e seus pesos...")
#salvando apenas o .h5
model.save("meuModelo.h5")


print("Carregando o modelo de um h5 já com pesos...")
model = keras.models.load_model("meuModelo.h5")


print("Avaliando o modelo...")

# Evaluate the model
loss, acc = model.evaluate(val_ds, verbose=2)
print("Model, accuracy: {:5.2f}%".format(100 * acc))


# Fazendo predições:

#cats
imagesCat = glob.glob("./PetImages/Cat/*.jpg")
imagesCat = [Image.open(random.choice(imagesCat)).resize((180, 180)) for i in range(10)]

#dogs
imagesDog = glob.glob("./PetImages/Dog/*.jpg")
imagesDog = [Image.open(random.choice(imagesDog)).resize((180, 180)) for i in range(10)]

lucas_images = []
im = Image.open("eu.jpg")
im = im.resize((180, 180))
lucas_images.append(im)
im = Image.open("eu2.jpg")
im = im.resize((180, 180))
lucas_images.append(im)

#Juntando as imagens
images = []
images.extend(imagesDog)
images.extend(imagesCat)
images.extend(lucas_images)


# Realizando a Predição apartir do modelo criado/carregado
out_np = [np.array(out) for out in images]
out_np_1 = [np.expand_dims(out, axis=0) for out in out_np] 
predicao = [model.predict(out) for out in out_np_1]
saida = ["Cat" if i[0][0] < 0.5 else "Dog" for i in predicao]
print(list(enumerate(saida)))