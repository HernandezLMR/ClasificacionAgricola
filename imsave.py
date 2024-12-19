import os
import cv2 as cv
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern, hog
from skimage.filters import gabor
from skimage.segmentation import slic
from skimage import io, color
from sklearn.decomposition import PCA
from skimage.feature import graycomatrix, graycoprops

# Ruta de las imágenes originales
input_path = "Agricultural-crops"
output_path = "Processed-Images"

# Crear carpeta de salida si no existe
os.makedirs(output_path, exist_ok=True)


# Paso 1: Preprocesamiento de las imágenes
def preprocess_images(input_path):
    """
    Preprocesa las imágenes: redimensiona, verifica que sean RGB y aplica filtro Gaussiano.
    Retorna un DataFrame con rutas y etiquetas.
    """
    data = []
    folders = os.listdir(input_path)

    for folder in folders:
        folder_path = os.path.join(input_path, folder)
        output_folder_path = os.path.join(output_path, folder)
        os.makedirs(output_folder_path, exist_ok=True)

        images = os.listdir(folder_path)
        for image_file in images:
            image_path = os.path.join(folder_path, image_file)
            output_image_path = os.path.join(output_folder_path, image_file)

            # Leer la imagen
            img = cv.imread(image_path)
            if img is None:
                continue

            # Convertir a RGB si no lo es
            if len(img.shape) == 2 or img.shape[-1] != 3:
                img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

            # Redimensionar a un tamaño uniforme (100x100)
            resized_img = cv.resize(img, (100, 100))

            # Aplicar filtro Gaussiano para suavizar
            smoothed_img = cv.GaussianBlur(resized_img, (5, 5), 0)

            # Guardar imagen preprocesada
            cv.imwrite(output_image_path, smoothed_img)

            # Agregar al DataFrame
            data.append([output_image_path, folder])

    # Crear DataFrame con rutas de imágenes y etiquetas
    df = pd.DataFrame(data, columns=["image", "label"])
    return df


# Paso 2: Extracción de características
def segment_image_slic(image_path):
    img = io.imread(image_path)
    if img is None:
        return None
    if len(img.shape) == 2 or img.shape[-1] != 3:
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    img = color.rgb2lab(img)
    segments = slic(img, n_segments=100, compactness=10, start_label=1)
    segmented_img = color.label2rgb(segments, img, kind="avg")
    resized = cv.resize(segmented_img, (100, 100))
    return resized


def segment_image_gabor(image_path):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        return None
    filtered_real, _ = gabor(img, frequency=0.6)
    resized = cv.resize(filtered_real, (100, 100))
    return resized


def segment_image_pca(image_path):
    img = cv.imread(image_path)
    if img is None:
        return None
    img_flat = img.reshape(-1, 3).astype(np.float32)
    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(img_flat)
    segmented = pca_result.reshape(img.shape[:2])
    segmented = cv.normalize(segmented, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    resized = cv.resize(segmented, (100, 100))
    return resized


def extract_sift_features(image_path):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        return None
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    if descriptors is not None:
        return descriptors.flatten()[:500]  # Limitar a 500 características
    return np.zeros(500)


def extract_lbp_features(image_path, radius=3, n_points=24):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        return None
    lbp = local_binary_pattern(img, n_points, radius, method="uniform")
    hist, _ = np.histogram(
        lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2)
    )
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-7
    return hist


def extract_hog_features(image_path):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        return None
    resized_img = cv.resize(img, (100, 100))
    features, _ = hog(
        resized_img,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True,
    )
    return features


def extract_glcm_features(image_path):
    """
    Calcula características basadas en la matriz de co-ocurrencia (GLCM).
    """
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Calcular la matriz de co-ocurrencia
    glcm = graycomatrix(
        img,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )

    # Extraer propiedades
    contrast = graycoprops(glcm, "contrast").mean()
    homogeneity = graycoprops(glcm, "homogeneity").mean()
    energy = graycoprops(glcm, "energy").mean()
    correlation = graycoprops(glcm, "correlation").mean()

    # Retornar las propiedades como un vector
    return np.array([contrast, homogeneity, energy, correlation])


# Paso 3: Procesar imágenes y guardar resultados
def main():
    df = preprocess_images(input_path)

    x_data = []
    y_data = []

    for idx, row in df.iterrows():
        label = row["label"]
        image_path = row["image"]

        slic_features = segment_image_slic(image_path)
        gabor_features = segment_image_gabor(image_path)
        pca_features = segment_image_pca(image_path)
        sift_features = extract_sift_features(image_path)
        lbp_features = extract_lbp_features(image_path)
        hog_features = extract_hog_features(image_path)
        glcm_features = extract_glcm_features(image_path)

        if (
            slic_features is not None
            and gabor_features is not None
            and pca_features is not None
            and sift_features is not None
            and lbp_features is not None
            and hog_features is not None
            and glcm_features is not None
        ):
            combined_features = np.concatenate(
                [
                    slic_features.flatten(),
                    gabor_features.flatten(),
                    pca_features.flatten(),
                    sift_features,
                    lbp_features,
                    hog_features,
                    glcm_features,
                ]
            )
            x_data.append(combined_features)
            y_data.append(label)

    # Guardar en NumPy
    np.save(os.path.join(output_path, "x_data.npy"), np.array(x_data))
    np.save(os.path.join(output_path, "y_data.npy"), np.array(y_data))
    print("Procesamiento completado. Datos guardados en la carpeta:", output_path)


if __name__ == "__main__":
    main()
