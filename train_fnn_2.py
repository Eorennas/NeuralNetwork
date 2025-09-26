import os, time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc, precision_score, recall_score, f1_score, log_loss
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ---------------------------
# Configurações
# ---------------------------
img_size = (128, 128)
batch_size = 32
dataset_dir = "dataset"
epochs = 50

os.makedirs("fnn_metrics", exist_ok=True)

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
# Converter generator em arrays
# ---------------------------
def generator_to_array(generator):
    X_list, y_list = [], []
    for i in range(len(generator)):
        X_batch, y_batch = generator[i]
        X_list.append(X_batch)
        y_list.append(y_batch)
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    return X, y

X_train, y_train = generator_to_array(train_gen)
X_val, y_val = generator_to_array(val_gen)

X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_val_flat = X_val.reshape((X_val.shape[0], -1))

# ---------------------------
# Pesos de classe
# ---------------------------
class_weights = compute_class_weight(class_weight="balanced",
                                     classes=np.unique(y_train),
                                     y=y_train)
class_weights_dict = dict(zip(np.unique(y_train), class_weights))
print("Pesos de classe:", class_weights_dict)

# ---------------------------
# Criar modelo FNN
# ---------------------------
model = models.Sequential([
    layers.Input(shape=(img_size[0]*img_size[1]*3,)),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# ---------------------------
# Callbacks
# ---------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("fnn_metrics/fnn_best_model.keras", monitor='val_loss', save_best_only=True)

# ---------------------------
# Treinamento
# ---------------------------
start_time = time.time()
history = model.fit(X_train_flat, y_train,
                    validation_data=(X_val_flat, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stop, checkpoint],
                    class_weight=class_weights_dict)
end_time = time.time()
print(f"Tempo de treino FNN: {end_time - start_time:.2f} segundos")

# ---------------------------
# Métricas
# ---------------------------
y_pred_prob = model.predict(X_val_flat)
y_pred = (y_pred_prob > 0.5).astype(int)

cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(train_gen.class_indices.keys()))
disp.plot()
plt.title("Matriz de Confusão - FNN")
plt.savefig("fnn_metrics/fnn_confusion_matrix.png")

fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0,1],[0,1],'--',color='gray')
plt.title("Curva ROC - FNN")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("fnn_metrics/fnn_roc_auc.png")

plt.figure()
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title("Acurácia - FNN")
plt.xlabel("Época")
plt.ylabel("Acurácia")
plt.legend()
plt.savefig("fnn_metrics/fnn_accuracy.png")

plt.figure()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title("Loss - FNN")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()
plt.savefig("fnn_metrics/fnn_loss.png")

precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
logloss = log_loss(y_val, y_pred_prob)
report = classification_report(y_val, y_pred, target_names=list(train_gen.class_indices.keys()))
with open("fnn_metrics/fnn_classification_report.txt", "w") as f:
    f.write(f"Inference time: {end_time - start_time:.2f} s\n")
    f.write(report)
    f.write(f"\nAUC: {roc_auc:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-score: {f1:.4f}\n")
    f.write(f"Log Loss: {logloss:.4f}\n")

print(report)
print(f"AUC: {roc_auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, LogLoss: {logloss:.4f}")

model.save("fnn_model_tf.keras")
print("Modelo FNN salvo com sucesso!")
