import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, roc_curve, auc,
                             precision_score, recall_score, f1_score, log_loss)
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# ===========================
# Configurações
# ===========================
img_size = (224, 224)
batch_size = 32
train_dir = "dataset/train"
valid_dir = "dataset/valid"
epochs = 50
output_dir = "fnn_metrics_optimized" # Usar uma variável para o diretório de saída

os.makedirs(output_dir, exist_ok=True)

# ===========================
# PASSO 1: Corrigir a Pipeline de Dados
# ===========================

# Gerador para dados de TREINO: com data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
)

# Gerador para dados de VALIDAÇÃO: APENAS reescala
validation_datagen = ImageDataGenerator(rescale=1./255)

# Função para achatar as imagens em tempo real
def flatten_generator(generator):
    for x_batch, y_batch in generator:
        yield x_batch.reshape(x_batch.shape[0], -1), y_batch

train_generator_base = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

validation_generator_base = validation_datagen.flow_from_directory(
    valid_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False # Essencial para que a ordem dos rótulos seja mantida
)

train_generator_flat = flatten_generator(train_generator_base)
validation_generator_flat = flatten_generator(validation_generator_base)

# ===========================
# PASSO 2: Criar modelo FNN com Regularização
# ===========================
model = models.Sequential([
    layers.Input(shape=(img_size[0] * img_size[1] * 3,)),
    layers.Dense(512),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    layers.Dense(256),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.summary()

optimizer = Adam(learning_rate=1e-4)

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# ===========================
# Callbacks
# ===========================
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint(os.path.join(output_dir, "fnn_best_model.keras"), monitor='val_loss', save_best_only=True)

# ===========================
# Treinamento com geradores
# ===========================
start_time = time.time()
history = model.fit(
    train_generator_flat,
    steps_per_epoch=train_generator_base.samples // batch_size,
    validation_data=validation_generator_flat,
    validation_steps=validation_generator_base.samples // batch_size,
    epochs=epochs,
    callbacks=[early_stop, checkpoint]
)
end_time = time.time()
print(f"Tempo de treino FNN Otimizado: {end_time - start_time:.2f} segundos")

# =======================================================
# SEÇÃO DE MÉTRICAS CORRIGIDA
# =======================================================
print("\n--- Iniciando Avaliação do Modelo ---")

# 1. Obter os rótulos verdadeiros (y_val) diretamente do gerador.
# Como shuffle=False, a ordem das classes corresponde aos arquivos.
y_val = validation_generator_base.classes

# 2. Fazer a predição usando o gerador de validação com os dados achatados.
# O model.predict também aceita geradores como entrada.
y_pred_prob = model.predict(flatten_generator(validation_generator_base),
                            steps=np.ceil(validation_generator_base.samples / batch_size))

# Se o número de amostras não for um múltiplo exato do batch_size,
# a predição pode retornar alguns exemplos a mais, então garantimos que os tamanhos batem.
if len(y_pred_prob) > len(y_val):
    y_pred_prob = y_pred_prob[:len(y_val)]

# 3. Converter probabilidades em classes (0 ou 1)
y_pred = (y_pred_prob > 0.5).astype(int)

# Nomes das classes para os gráficos e relatórios
class_names = list(validation_generator_base.class_indices.keys())

# Matriz de confusão
cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão - FNN Otimizado")
plt.savefig(os.path.join(output_dir, "fnn_confusion_matrix.png"))
print("Matriz de confusão salva.")

# Curva ROC e AUC
fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('Curva ROC - FNN Otimizado')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.legend(loc="lower right")
plt.savefig(os.path.join(output_dir, "fnn_roc_auc.png"))
print("Curva ROC salva.")

# Gráficos de acurácia e loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Acurácia de Treino')
plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
plt.title('Acurácia do Modelo')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss de Treino')
plt.plot(history.history['val_loss'], label='Loss de Validação')
plt.title('Loss do Modelo')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(output_dir, "fnn_history_plots.png"))
print("Gráficos de histórico salvos.")


# Relatório detalhado
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
logloss = log_loss(y_val, y_pred_prob)
report = classification_report(y_val, y_pred, target_names=class_names)

report_text = f"Tempo de Treino: {end_time - start_time:.2f} segundos\n\n"
report_text += "--- Relatório de Classificação ---\n"
report_text += report
report_text += "\n--- Métricas Adicionais ---\n"
report_text += f"AUC: {roc_auc:.4f}\n"
report_text += f"Precision (macro): {precision:.4f}\n"
report_text += f"Recall (macro): {recall:.4f}\n"
report_text += f"F1-score (macro): {f1:.4f}\n"
report_text += f"Log Loss: {logloss:.4f}\n"

with open(os.path.join(output_dir, "fnn_classification_report.txt"), "w") as f:
    f.write(report_text)

print(report_text)
print("Relatório de classificação salvo.")

# ===========================
# Salvar modelo
# ===========================
model.save(os.path.join(output_dir, "fnn_model_final.keras"))
print("Modelo FNN salvo com sucesso!")