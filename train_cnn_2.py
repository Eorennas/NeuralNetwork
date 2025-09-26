import os, time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------
# Configurações
# ---------------------------
img_size = (128, 128)
batch_size = 32
dataset_dir = "dataset"
epochs = 50

os.makedirs("cnn_metrics", exist_ok=True)

# ---------------------------
# Carregar dataset
# ---------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
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

val_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# ---------------------------
# Pesos de classe para balancear
# ---------------------------
y_train = train_gen.classes
class_weights = compute_class_weight(class_weight="balanced",
                                     classes=np.unique(y_train),
                                     y=y_train)
class_weights_dict = dict(zip(np.unique(y_train), class_weights))
print("Pesos de classe:", class_weights_dict)

# ---------------------------
# Criar CNN
# ---------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# ---------------------------
# EarlyStopping
# ---------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ---------------------------
# Treinamento
# ---------------------------
start_time = time.time()
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=[early_stop],
    class_weight=class_weights_dict
)
end_time = time.time()
print(f"Tempo de treino CNN: {end_time - start_time:.2f} segundos")

# ---------------------------
# Avaliação
# ---------------------------
y_val = val_gen.classes
y_pred_prob = model.predict(val_gen)
y_pred = (y_pred_prob > 0.5).astype(int)

# Matriz de Confusão
cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(train_gen.class_indices.keys()))
disp.plot()
plt.title("Matriz de Confusão - CNN")
plt.savefig("cnn_metrics/cnn_confusion_matrix.png")

# ROC e AUC
fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0,1],[0,1],'--',color='gray')
plt.title("Curva ROC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("cnn_metrics/cnn_roc_auc.png")

# Accuracy e Loss
plt.figure()
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title("Acurácia")
plt.xlabel("Época")
plt.ylabel("Acurácia")
plt.legend()
plt.savefig("cnn_metrics/cnn_accuracy.png")

plt.figure()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title("Loss")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()
plt.savefig("cnn_metrics/cnn_loss.png")

# Classification report detalhado
report = classification_report(y_val, y_pred, target_names=list(train_gen.class_indices.keys()))
print(report)

# Salvar modelo
model.save("cnn_model_tf.keras")
print("Modelo CNN salvo com sucesso!")
