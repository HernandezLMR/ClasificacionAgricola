import os
import cv2 as cv
import numpy as np
import pandas as pd
from skimage import io, color
from skimage.filters import gabor
from skimage.segmentation import slic
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch
import torchvision.models as models
import torchvision.transforms as transforms

input_path = "Agricultural-crops"
output_path = "Processed-Images"

os.makedirs(output_path, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50 = models.resnet50(pretrained=True).to(device)
resnet50.eval()


features_hook = []


def hook(module, input, output):
    features_hook.append(output)


resnet50.avgpool.register_forward_hook(hook)


transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def preprocess_images(input_path):
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

            img = cv.imread(image_path)
            if img is None:
                continue


            if len(img.shape) == 2 or img.shape[-1] != 3:
                img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)


            resized_img = cv.resize(img, (100, 100))


            smoothed_img = cv.GaussianBlur(resized_img, (5, 5), 0)


            cv.imwrite(output_image_path, smoothed_img)


            data.append([output_image_path, folder])


    df = pd.DataFrame(data, columns=["image", "label"])
    return df



def extract_resnet50_features(image_path):
    img = cv.imread(image_path)
    if img is None:
        return None

    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)


    global features_hook
    features_hook = []


    with torch.no_grad():
        _ = resnet50(img_tensor)


    if features_hook:
        features = features_hook[0].squeeze().cpu().numpy()
        return features
    return None



def main():
    df = preprocess_images(input_path)

    x_data = []
    y_data = []

    for idx, row in df.iterrows():
        label = row["label"]
        image_path = row["image"]

        resnet_features = extract_resnet50_features(image_path)

        if resnet_features is not None:
            x_data.append(resnet_features)
            y_data.append(label)

    np.save(os.path.join(output_path, "x_data.npy"), np.array(x_data))
    np.save(os.path.join(output_path, "y_data.npy"), np.array(y_data))

    print("Procesamiento completado. Datos guardados en la carpeta:", output_path)


if __name__ == "__main__":
    main()
