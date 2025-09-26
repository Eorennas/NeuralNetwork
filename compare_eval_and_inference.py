import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, roc_curve, auc,
                             precision_score, recall_score, f1_score, log_loss,
                             accuracy_score)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.patches as mpatches


# ===========================
# Configurações
# ===========================
img_size = (128, 128)
threshold = 0.5
real_test_dir = "real_test"  # pasta com subpastas 'pessoa/' e 'nao_pessoa/'
cnn_model_path = "cnn_model_tf.keras"
fnn_model_path = "fnn_model_tf.keras"

os.makedirs("result_compare", exist_ok=True)
class_indices = {"nao_pessoa": 0, "pessoa": 1}

# ===========================
# Carregar modelos
# ===========================
cnn_model = load_model(cnn_model_path)
fnn_model = load_model(fnn_model_path)

# ===========================
# Carregar imagens de teste
# ===========================
X_test_list, y_test_list = [], []

for label in class_indices.keys():
    class_dir = os.path.join(real_test_dir, label)
    for fname in os.listdir(class_dir):
        fpath = os.path.join(class_dir, fname)
        img = image.load_img(fpath, target_size=img_size)
        img_array = image.img_to_array(img) / 255.0
        X_test_list.append(img_array)
        y_test_list.append(class_indices[label])

X_test = np.array(X_test_list)
y_test = np.array(y_test_list)
X_test_flat = X_test.reshape((X_test.shape[0], -1))  # para FNN

# ===========================
# Função de avaliação
# ===========================
def evaluate_model(model, X, y, model_name):
    start_time = time.time()
    y_pred_prob = model.predict(X)
    end_time = time.time()
    inference_time = end_time - start_time

    # Predições binárias
    y_pred = (y_pred_prob > threshold).astype(int)

    # Métricas detalhadas
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    logloss = log_loss(y, y_pred_prob)
    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    report = classification_report(y, y_pred, target_names=list(class_indices.keys()))

    # Salvar relatório completo
    with open(f"result_compare/{model_name}_metrics.txt", "w") as f:
        f.write(f"Modelo: {model_name}\n")
        f.write(f"Inference time total: {inference_time:.4f} s\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"Log Loss: {logloss:.4f}\n")
        f.write(f"AUC: {roc_auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Matriz de Confusão
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=list(class_indices.keys()))
    disp.plot()
    plt.title(f"Matriz de Confusão - {model_name}")
    plt.savefig(f"result_compare/{model_name}_confusion_matrix.png")
    plt.close()

    # Curva ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.title(f"Curva ROC - {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(f"result_compare/{model_name}_roc_auc.png")
    plt.close()

    print(f"{model_name} avaliado! Métricas salvas em 'result_compare/{model_name}_metrics.txt'.")

    return y_pred

# ===========================
# Avaliar CNN e FNN
# ===========================
y_pred_cnn = evaluate_model(cnn_model, X_test, y_test, "CNN")
y_pred_fnn = evaluate_model(fnn_model, X_test_flat, y_test, "FNN")

# ===========================
# Mapa visual com porcentagens
# ===========================
n_images = len(X_test)
cols = 5
rows = math.ceil(n_images / cols)

correct_both = 0
correct_cnn_only = 0
correct_fnn_only = 0
wrong_both = 0

plt.figure(figsize=(cols * 3, rows * 3))
for i in range(n_images):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(X_test[i].astype("float32"))
    plt.axis('off')

    cnn_correct = (y_pred_cnn[i][0] == y_test[i]) if y_pred_cnn.ndim > 1 else (y_pred_cnn[i] == y_test[i])
    fnn_correct = (y_pred_fnn[i][0] == y_test[i]) if y_pred_fnn.ndim > 1 else (y_pred_fnn[i] == y_test[i])

    if cnn_correct and fnn_correct:
        color = 'green'
        correct_both += 1
    elif cnn_correct and not fnn_correct:
        color = 'black'
        correct_cnn_only += 1
    elif not cnn_correct and fnn_correct:
        color = 'blue'
        correct_fnn_only += 1
    else:
        color = 'red'
        wrong_both += 1

    cnn_label = list(class_indices.keys())[y_pred_cnn[i][0]] if y_pred_cnn.ndim > 1 else list(class_indices.keys())[y_pred_cnn[i]]
    fnn_label = list(class_indices.keys())[y_pred_fnn[i][0]] if y_pred_fnn.ndim > 1 else list(class_indices.keys())[y_pred_fnn[i]]

    plt.title(f"CNN:{cnn_label}\nFNN:{fnn_label}", color=color, fontsize=8)

# Legenda com porcentagens
total = n_images
patches = [
    mpatches.Patch(color='green', label=f'CNN+FNN corretos ({correct_both/total*100:.1f}%)'),
    mpatches.Patch(color='black', label=f'Só CNN correto ({correct_cnn_only/total*100:.1f}%)'),
    mpatches.Patch(color='blue', label=f'Só FNN correto ({correct_fnn_only/total*100:.1f}%)'),
    mpatches.Patch(color='red', label=f'Ambos errados ({wrong_both/total*100:.1f}%)')
]
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("result_compare/comparison_image_map_colored.png")
plt.close()

print("Mapa visual salvo em 'result_compare/comparison_image_map_colored.png'.")
