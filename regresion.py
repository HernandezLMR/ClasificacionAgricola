import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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


# Paso 2: Escalado de los datos
def scale_data(x_train, x_test):
    """
    Escala los datos para normalizar las características.
    """
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled

def plot_confusion_matrix(y_test, y_pred, labels):
    """
    Plot confusion matrix with improved readability of x-axis.
    """
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    # Create confusion matrix display
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))  # Increase figure size
    cm_display.plot(cmap=plt.cm.Blues, ax=ax)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")
    
    # Adjust layout to prevent clipping of labels
    plt.tight_layout()
    
    # Show the plot
    plt.show()
# Paso 3: Entrenamiento y evaluación del modelo Logistic Regression
def train_logistic_regression(x_data, y_data):
    """
    Train a logistic regression model and evaluate it.
    """
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    # Scale data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Train the model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(x_test_scaled)

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, labels=model.classes_)


# Paso 4: Guardar el modelo entrenado (opcional)
def save_model(model, file_path):
    """
    Guarda el modelo entrenado en un archivo pickle.
    """
    import joblib

    joblib.dump(model, file_path)
    print(f"Modelo guardado en: {file_path}")


# Paso 5: Pipeline principal
def main():
    # Cargar los datos
    x_data, y_data = load_data(x_data_file, y_data_file)

    # Entrenar y evaluar el modelo
    model = train_logistic_regression(x_data, y_data)

    # Guardar el modelo entrenado
    save_model(model, f"{data_path}/logistic_regression_model.pkl")


if __name__ == "__main__":
    main()
