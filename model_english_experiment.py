import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall

# CARGA DE DATOS
data_path = "SaberIngles.csv"  # Ruta del archivo dentro del repositorio
df_Ingles = pd.read_csv(data_path)
df_Ingles.head()

# PREPARACIÓN DE DATOS
variables_seleccionadas = ["naturaleza", "Nivelmate", "calendario", "genero"]
variables_nominales = ["naturaleza", "Nivelmate", "calendario", "genero"]
# Aplicar One-Hot Encoding
one_hot_encoder = OneHotEncoder(sparse_output=False, drop="first")  # Evita colinealidad
X_transformed = one_hot_encoder.fit_transform(df_Ingles[variables_nominales])

# Crear y aplicar LabelEncoder
le = LabelEncoder()
y_int = le.fit_transform(df_Ingles["nivelingles"])

# BALANCEO DE CLASES
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
# Estrategia de balanceo
resampling_pipeline = Pipeline([
    # Elimina parte de la clase mayoritaria (por defecto 'majority')
    ('undersample', RandomUnderSampler(sampling_strategy='majority', random_state=42)),
    
    # Luego clona las clases restantes para igualarlas (menos la ya reducida)
    ('oversample', RandomOverSampler(sampling_strategy='not majority', random_state=42))
])
#Under y over sampling
#X_resampled, y_resampled_int = resampling_pipeline.fit_resample(X_transformed, y_int)

#Solo over sampling
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled_int = oversampler.fit_resample(X_transformed, y_int)

# Volver a convertir etiquetas a one-hot
y_resampled = tf.keras.utils.to_categorical(y_resampled_int, num_classes=5)

# División de datos en entrenamiento, validación y pruebas
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42)

tf.keras.backend.clear_session()

# CONFIGURAR MLflow
#experiment_name = "Model_Englishh"
#experiment_id = mlflow.create_experiment(experiment_name)
#experiment = mlflow.set_experiment("Model_Englishh")
mlflow.set_tracking_uri("http://localhost:5000")  # Cambia la URL según tu configuración

with mlflow.start_run(experiment_id=236681886414905201): 
    # CONSTRUCCIÓN DEL MODELO
    capas = [
        {"tipo": "Dense", "unidades": 128, "activacion": "ReLU", "dropout": 0.3},
        {"tipo": "Dense", "unidades": 64, "activacion": "ReLU", "dropout": 0.2},
        {"tipo": "Dense", "unidades": 5, "activacion": "softmax"}
    ]
    
    model_english = keras.Sequential([
         keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(5, activation='softmax')
    ])

    #from tensorflow.keras.optimizers import SGD
    #optimizer = SGD(learning_rate=0.01, momentum=0.9)
    optimizer = keras.optimizers.Adam(learning_rate=0.001) #en vez del valor por defecto (0.001).
    
    model_english.compile(loss="categorical_crossentropy", optimizer=optimizer,
                       metrics=[CategoricalAccuracy(), keras.metrics.Precision(), keras.metrics.Recall()])
    
    mlflow.log_param("Optimizador", type(optimizer).__name__)  # Ej: 'Adam'
    mlflow.log_param("Tasa_Aprendizaje", optimizer.learning_rate.numpy())

    # REGISTRO DE VARIABLES Y PARÁMETROS EN MLflow
    mlflow.log_param("Variables_Seleccionadas", str(variables_seleccionadas))
    mlflow.log_param("Num_Capas", len(capas))

    #callback = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)
    ]
    mlflow.log_param("Callback_EarlyStopping_and_ReduceLR", "Si")
    #mlflow.log_param("EarlyStopping_Patience", callback.patience)

    for i, capa in enumerate(capas):
        mlflow.log_param(f"Capa_{i+1}_Tipo", capa["tipo"])
        mlflow.log_param(f"Capa_{i+1}_Unidades", capa.get("unidades", ""))
        mlflow.log_param(f"Capa_{i+1}_Activacion", capa.get("activacion", ""))
        if "dropout" in capa:
            mlflow.log_param(f"Capa_{i+1}_Dropout", capa["dropout"])

    # ENTRENAMIENTO DEL MODELO
    #history = model_english.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_valid, y_valid), verbose=0)
    history = model_english.fit(X_train, y_train,
                    validation_data=(X_valid, y_valid),
                    epochs=50,
                    batch_size=32,
                    callbacks=callbacks, verbose=0)
    
    model_english.save("modelo_ingles_entrenado.keras")
    mlflow.log_artifact("modelo_ingles_entrenado.keras")

    from mlflow.models import infer_signature 
    infer_signature(model_input=X_transformed[0])
    
    # EVALUACIÓN DEL MODELO
    
    #PREDICCION
    predicciones = model_english.predict(X_test)  #Devuelve la probabilidad de cada clase (n_muestras, 6)
    # Convertir los datos de prueba y las predicciones a etiquetas de clase
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(predicciones, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro") #Usamos macro porque tenemos clases desbalanceadas
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    # Calcular la pérdida
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    loss_value = loss_fn(y_test, predicciones).numpy()

    mlflow.log_metric("Test_Loss", loss_value)
    mlflow.log_metric("Test_Accuracy", accuracy)
    mlflow.log_metric("Test_Precision", precision)
    mlflow.log_metric("Test_Recall", recall)
    mlflow.log_metric("Test_F1-score", f1)

    # MATRIZ DE CONFUSIÓN
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Guardar la matriz de confusión como imagen
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta Real")
    plt.title("Matriz de Confusión")
    plt.savefig("confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("confusion_matrix.png")

    def plot_training_history(history):
        # Extraer datos del history
        acc = history.history.get('accuracy') or history.history.get('categorical_accuracy')
        val_acc = history.history.get('val_accuracy') or history.history.get('val_categorical_accuracy')
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        precision = history.history.get('precision')
        val_precision = history.history.get('val_precision')
        recall = history.history.get('recall')
        val_recall = history.history.get('val_recall')

        epochs = range(1, len(acc) + 1)


        # Configurar la figura
        plt.figure(figsize=(14, 5))

        # Subplot 1: Precisión
        plt.subplot(1, 3, 1)
        plt.plot(epochs, acc, 'bo-', label='Entrenamiento')
        plt.plot(epochs, val_acc, 'ro-', label='Validación')
        plt.title('Precisión durante el entrenamiento')
        plt.xlabel('Épocas')
        plt.ylabel('Precisión')
        plt.legend()
        plt.grid(True)

        # Subplot 2: Pérdida
        plt.subplot(1, 3, 2)
        plt.plot(epochs, loss, 'bo-', label='Entrenamiento')
        plt.plot(epochs, val_loss, 'ro-', label='Validación')
        plt.title('Pérdida durante el entrenamiento')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.grid(True)

        # Subplot 3: Precision y Recall
        plt.subplot(1, 3, 3)
        if precision and recall:
            plt.plot(epochs, precision, 'go-', label='Precision (Entrenamiento)')
            plt.plot(epochs, recall, 'mo-', label='Recall (Entrenamiento)')
        if val_precision and val_recall:
            plt.plot(epochs, val_precision, 'g--', label='Precision (Validación)')
            plt.plot(epochs, val_recall, 'm--', label='Recall (Validación)')
        plt.title('Precision y Recall por época')
        plt.xlabel('Épocas')
        plt.ylabel('Valor')

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("Metrics_by_epoch.png")
        plt.close()

    plot_training_history(history)
    mlflow.log_artifact("Metrics_by_epoch.png")


    # Guardar el modelo en MLflow
    mlflow.sklearn.log_model(model_english, "modelo_ingles")

    print("Experimento registrado en MLflow.")