import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Ruta de los datos preprocesados
data_path = "Processed-Images"
x_data_file = f"{data_path}/x_data.npy"
y_data_file = f"{data_path}/y_data.npy"


# Paso 1: Cargar los datos
def load_data(x_file, y_file):
    """
    Carga los datos preprocesados desde archivos NumPy.
    """
    x_data = np.load(x_file)
    y_data = np.load(y_file)
    return x_data, y_data


# Paso 2: Entrenamiento y evaluación del modelo Decision Tree
def train_decision_tree(x_data, y_data):
    """
    Entrena y evalúa un modelo Decision Tree.
    """
    # Dividir en conjuntos de entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42
    )

    # Entrenar el modelo
    model = DecisionTreeClassifier(
        random_state=42, max_depth=10
    )  # Ajustar max_depth si es necesario
    model.fit(x_train, y_train)

    # Evaluar el modelo
    y_pred = model.predict(x_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return model


# Paso 3: Guardar el modelo entrenado (opcional)
def save_model(model, file_path):
    """
    Guarda el modelo entrenado en un archivo pickle.
    """
    import joblib

    joblib.dump(model, file_path)
    print(f"Modelo guardado en: {file_path}")


# Paso 4: Pipeline principal
def main():
    # Cargar los datos
    x_data, y_data = load_data(x_data_file, y_data_file)

    # Entrenar y evaluar el modelo
    model = train_decision_tree(x_data, y_data)

    # Guardar el modelo entrenado
    save_model(model, f"{data_path}/decision_tree_model.pkl")


if __name__ == "__main__":
    main()
