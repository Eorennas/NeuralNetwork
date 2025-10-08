import os
import time
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# --- 1. Parâmetros ---
img_size = (224, 224)
batch_size = 32
epochs = 50

train_dir = "dataset/train"
validation_dir = "dataset/valid"
output_dir = "mobilenet_metrics" 

os.makedirs(output_dir, exist_ok=True)

# --- 2. Geradores de Dados ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    directory=validation_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# --- 3. Construção do Modelo (Transfer Learning) ---
base_model = MobileNetV2(input_shape=img_size + (3,),
                         include_top=False,
                         weights='imagenet')
# Congela as camadas da base_model
base_model.trainable = False

# Adiciona um cabeçalho de classificação personalizado
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x) # Adicionado Dropout para regularização
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compila o modelo com mais métricas
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# --- Custo Computacional (Parâmetros do Modelo) ---
summary_str = []
model.summary(print_fn=lambda x: summary_str.append(x))
summary_text = "\n".join(summary_str)
with open(os.path.join(output_dir, "mobilenet_model_summary.txt"), "w") as f:
    f.write(summary_text)
print("Sumário do modelo salvo.")
print(summary_text)


# --- 4. Callbacks para um Treinamento Melhor ---
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
checkpoint = ModelCheckpoint(
    filepath=os.path.join(output_dir, "mobilenet_best_model.keras"),
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# --- 5. Treinar o Modelo ---
print("\nIniciando o treinamento do modelo...")
start_time = time.time()
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs,
    callbacks=[early_stop, lr_scheduler, checkpoint] 
)
end_time = time.time()

# --- Tempo de Treinamento ---
training_time = end_time - start_time
print(f"\nTempo de treino: {training_time:.2f} segundos")
with open(os.path.join(output_dir, "mobilenet_training_time.txt"), "w") as f:
    f.write(f"Tempo total de treinamento: {training_time:.2f} segundos")


# --- 6. Salvar Modelo Final ---
model.save('mobilenet_final_model.keras')
print(f"\nModelo final salvo em 'mobilenet_final_model.keras'")


# --- 7. Análise e Visualização de Métricas ---
print("\nGerando gráficos e métricas de análise...")

# Gráficos de Acurácia e Loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Acurácia de Treino')
plt.plot(epochs_range, val_acc, label='Acurácia de Validação')
plt.legend(loc='lower right')
plt.title('Acurácia de Treino e Validação')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Loss de Treino')
plt.plot(epochs_range, val_loss, label='Loss de Validação')
plt.legend(loc='upper right')
plt.title('Loss de Treino e Validação')
plt.xlabel('Épocas')
plt.ylabel('Loss')

plt.suptitle('Análise do Histórico de Treinamento', fontsize=16)
plt.savefig(os.path.join(output_dir, 'grafico_acuracia_loss.png'))
#plt.show()

# Predições no conjunto de validação para as métricas
y_pred_probs = model.predict(validation_generator, steps=np.ceil(validation_generator.samples / batch_size))
y_pred = (y_pred_probs > 0.5).astype(int)
y_true = validation_generator.classes[:len(y_pred)] # Garante que os tamanhos sejam iguais
class_labels = list(validation_generator.class_indices.keys())

# Matriz de Confusão
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Matriz de Confusão')
plt.ylabel('Rótulo Verdadeiro')
plt.xlabel('Rótulo Previsto')
plt.savefig(os.path.join(output_dir, 'matriz_confusao.png'))
#plt.show()

# Relatório de Classificação (Precisão, Recall, F1-Score)
print("\n--- Relatório de Classificação ---")
report = classification_report(y_true, y_pred, target_names=class_labels)
print(report)
with open(os.path.join(output_dir, 'relatorio_classificacao.txt'), 'w') as f:
    f.write(report)

# Curva ROC e AUC
fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.savefig(os.path.join(output_dir, 'curva_roc.png'))
#plt.show()

print(f"\nMétricas e gráficos salvos na pasta '{output_dir}'.")