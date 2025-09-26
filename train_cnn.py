import os, time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # Importar ReduceLROnPlateau

# ---------------------------
# Configurações
# ---------------------------
img_size = (128, 128)
batch_size = 32
dataset_dir = "dataset"
epochs = 50 # Aumentar as épocas, já que temos EarlyStopping

os.makedirs("cnn_metrics_v2", exist_ok=True)

# ---------------------------
# Carregar dataset com Augmentation mais robusto
# ---------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    shuffle=True
)

# A validação NÃO DEVE ter augmentation, apenas rescale
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_gen = val_datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# ---------------------------
# Criar CNN Aprimorada com BatchNormalization
# ---------------------------
model = models.Sequential([
    # Input Layer
    layers.Input(shape=(img_size[0], img_size[1], 3)),

    # Bloco 1
    layers.Conv2D(32, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2,2),

    # Bloco 2
    layers.Conv2D(64, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2, 2),

    # Bloco 3
    layers.Conv2D(128, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2, 2),
    
    # Adicionando um bloco extra para mais profundidade
    layers.Conv2D(256, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2, 2),

    # Classificador
    layers.Flatten(),
    layers.Dense(128),
    layers.BatchNormalization(), # Normalização também na parte densa
    layers.Activation('relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

model.summary() # Bom para visualizar a nova arquitetura

# ---------------------------
# Callbacks Aprimorados
# ---------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10, # Aumentar a paciência, pois o LR scheduler pode causar platôs temporários
    restore_best_weights=True,
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# ---------------------------
# Treinamento
# ---------------------------
start_time = time.time()
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=[early_stop, lr_scheduler] # Adiciona os dois callbacks
)
end_time = time.time()

print(f"Tempo de treino CNN: {end_time - start_time:.2f} segundos")

# ---------------------------
# O restante do seu código de avaliação (está perfeito)
# ... (matriz de confusão, ROC, gráficos, etc.)
# Lembre de salvar as novas métricas em "cnn_metrics_v2/"
# ---------------------------