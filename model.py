import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import zipfile
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, preprocessing, layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def extract_zip(pathToZip, pathToExtract):
  with zipfile.ZipFile(pathToZip, 'r') as zip_ref:
    zip_ref.extractall(pathToExtract)
  print("Zip file extracted successfully")

def load_images_and_labels(is_train=True):
    images = []
    labels = []

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

    base_dir = os.path.dirname(__file__)
    if is_train:
        non_humans_dir = os.path.join(base_dir, 'human-and-non-human/training_set/training_set/non-humans')
        humans_dir = os.path.join(base_dir, 'human-and-non-human/training_set/training_set/humans')
    else:
        non_humans_dir = os.path.join(base_dir, 'human-and-non-human/test_set/test_set/non-humans')
        humans_dir = os.path.join(base_dir, 'human-and-non-human/test_set/test_set/humans')

    for filename in os.listdir(non_humans_dir):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(non_humans_dir, filename)
            image = Image.open(image_path).convert('RGB').resize((128, 128))
            images.append(np.array(image))
            labels.append(0)

    for filename in os.listdir(humans_dir):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(humans_dir, filename)
            image = Image.open(image_path).convert('RGB').resize((128, 128))
            images.append(np.array(image))
            labels.append(1)

    print("Images and labels loaded successfully")

    return np.array(images), np.array(labels)


def normalize(images, labels):
  images = images.astype('float32') / 255.0
  X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
  print("Data normalized successfully")
  return X_train, X_test, y_train, y_test

def augment_data():
  datagen = preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
  )
  print("Data augmented successfully")
  return datagen

def create_model():
  model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
  ])
  print("Model created successfully")
  return model


def compile_model(model):
  model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
  )
  print("Model compiled successfully")

def fit_model(model, X_train, y_train, X_test, y_test):
  early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
  )

  datagen = augment_data()
  datagen.fit(X_train)
  history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    steps_per_epoch=32,
    epochs=50,
    callbacks=[early_stopping]
  )

  print("Model fitted successfully")

  return history

def show_graphs(history):
  plt.figure(figsize=(12, 4))

  plt.subplot(1, 2, 1)
  plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
  plt.plot(history.history['val_loss'], label='Pérdida de Validación')
  plt.title('Pérdida durante el Entrenamiento')
  plt.xlabel('Épocas')
  plt.ylabel('Pérdida')
  plt.legend()

# Graficar la precisión
  plt.subplot(1, 2, 2)
  plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
  plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
  plt.title('Precisión durante el Entrenamiento')
  plt.xlabel('Épocas')
  plt.ylabel('Precisión')
  plt.legend()

  plt.show()
  print("Graphs shown successfully")

if __name__ == '__main__':
  #extract_zip('./people-dataset.zip', './')
  images, labels = load_images_and_labels()
  X_train, X_test, y_train, y_test = normalize(images, labels)
  model = create_model()
  compile_model(model)
  history = fit_model(model, X_train, y_train, X_test, y_test)
  model.save('modelV4.h5')
  show_graphs(history)
