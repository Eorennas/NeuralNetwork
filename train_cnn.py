import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# ---------------------------
# Configurações
# ---------------------------
img_size = (224, 224)
batch_size = 32
dataset_dir = "dataset/train"
validat_dir = "dataset/valid"
epochs = 50
output_dir = "cnn_metrics"

os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# Carregar dataset
# ---------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    validat_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# ---------------------------
# Criar Modelo
# ---------------------------
model = models.Sequential([
    # Bloco 1
    layers.Conv2D(32, (3, 3), input_shape=(img_size[0], img_size[1], 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(64, (3, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Bloco 2
    layers.Conv2D(64, (3, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Bloco 3
    layers.Conv2D(128, (3, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Camadas densas
    layers.Flatten(),
    layers.Dense(64),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    # Acurácia, Precisão e Recall já são monitoradas durante o treino
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# --- Custo Computacional (Parâmetros do Modelo) ---
summary_str = []
model.summary(print_fn=lambda x: summary_str.append(x))
summary_text = "\n".join(summary_str)
with open(os.path.join(output_dir, "cnn_model_summary.txt"), "w") as f:
    f.write(summary_text)
print("Sumário do modelo salvo em cnn_model_summary.txt")
print(summary_text)

# ---------------------------
# Callbacks
# ---------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
checkpoint = ModelCheckpoint(os.path.join(output_dir, "cnn_best_model.keras"), monitor='val_loss', save_best_only=True, verbose=1)

# Pesos de classe para lidar com desbalanceamento
class_weights = {0: 1, 1: 1.36}

# ---------------------------
# Treinamento
# ---------------------------
start_time = time.time()
history = model.fit(
    train_gen,
    validation_data=val_gen,
    class_weight=class_weights,
    epochs=epochs,
    callbacks=[early_stop, lr_scheduler, checkpoint]
)
end_time = time.time()

# --- Tempo de Treinamento ---
training_time = end_time - start_time
print(f"Tempo de treino CNN: {training_time:.2f} segundos")
with open(os.path.join(output_dir, "cnn_training_time.txt"), "w") as f:
    f.write(f"Tempo total de treinamento: {training_time:.2f} segundos")

# ---------------------------
# Avaliação e Métricas
# ---------------------------
y_val = val_gen.classes
y_pred_prob = model.predict(val_gen)
y_pred = (y_pred_prob > 0.5).astype(int)

# --- Matriz de Confusão ---
cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(train_gen.class_indices.keys()))
disp.plot()
plt.title("Matriz de Confusão - CNN")
plt.savefig(os.path.join(output_dir, "cnn_confusion_matrix.png"))
print("Matriz de confusão salva.")

# --- Curva ROC e AUC ---
fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"Curva ROC (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("Curva ROC")
plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Verdadeiros Positivos")
plt.legend()
plt.savefig(os.path.join(output_dir, "cnn_roc_auc.png"))
print("Curva ROC e AUC salvos.")

# ---------------------------
# Gráficos de Acurácia e Loss
# ---------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Gráfico de Acurácia
ax1.plot(history.history['accuracy'], label='Acurácia de Treino')
ax1.plot(history.history['val_accuracy'], label='Acurácia de Validação')
ax1.set_title("Evolução da Acurácia")
ax1.set_xlabel("Época")
ax1.set_ylabel("Acurácia")
ax1.legend()

ax2.plot(history.history['loss'], label='Loss de Treino')
ax2.plot(history.history['val_loss'], label='Loss de Validação')
ax2.set_title("Evolução do Loss")
ax2.set_xlabel("Época")
ax2.set_ylabel("Loss")
ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cnn_accuracy_loss.png"))
print("Gráficos de acurácia e loss salvos em uma única imagem.")

# --- Relatório de Classificação (Precisão, Recall, F1-Score) ---
report = classification_report(y_val, y_pred, target_names=list(train_gen.class_indices.keys()))
print("\n--- Relatório de Classificação ---")
print(report)

# Salva o relatório em um arquivo
with open(os.path.join(output_dir, "cnn_classification_report.txt"), "w") as f:
    f.write(report)
print("Relatório de classificação salvo.")

# ---------------------------
# Salvar modelo final
# ---------------------------
model.save("cnn_model_tf.keras")
print("Modelo final salvo com sucesso em cnn_model_tf.keras!")