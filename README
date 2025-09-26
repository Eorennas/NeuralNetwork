Passo 1 – Instalar dependências
    pip install -r requirements.txt

Passo 2 - Preparar os datasets
    Certifique-se de que você tem as pastas organizadas:

    projeto/
    │
    ├─ dataset/
    │   ├─ pessoa/
    │   └─ nao_pessoa/
    │
    ├─ real_test/
    │   ├─ pessoa/
    │   └─ nao_pessoa/

Passo 3 – Treinar a CNN
    Execute o script da CNN:
     python treinar_cnn.py
    Vai treinar a CNN.
    Vai salvar métricas em cnn_metrics/.
    Vai salvar o modelo em cnn_model_tf.keras.

Passo 4 – Treinar a FNN
    Execute o script da FNN:
        python treinar_fnn.py
    Vai treinar a FNN usando as imagens achatadas.
    Vai salvar métricas em fnn_metrics/.
    Vai salvar o modelo em fnn_model_tf.keras.

Passo 5 – COMPARAR CNN e FNN
    Execute o script de teste com métricas detalhadas e mapa visual:
        python compare_eval_and_inference.py
    Vai carregar os modelos treinados (cnn_model_tf.keras e fnn_model_tf.keras)
    Vai rodar todas as métricas: Accuracy, Precision, Recall, F1, Log Loss, AUC
    Vai gerar confusion matrix e ROC curve
    Vai gerar mapa visual com cores e porcentagens em result_compare/comparison_image_map_colored.png
    Vai salvar relatório detalhado em result_compare/.

Passo 6 – Conferir resultados
    cnn_metrics/ → métricas e gráficos da CNN
    fnn_metrics/ → métricas e gráficos da FNN
    result_compare/ → comparação CNN vs FNN com mapa visual e métricas de teste
