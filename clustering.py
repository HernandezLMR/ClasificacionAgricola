import numpy as np
import cv2 as cv

def kmeans(image):
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    K = 2  # Number of clusters
    _, labels, centers = cv.kmeans(
        Z, 
        K, 
        None, 
        (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0), 
        10, 
        cv.KMEANS_RANDOM_CENTERS
    )
    
    # Create a binary image: 0 -> black, 1 -> white
    binary_image = labels.flatten().reshape(image.shape[:2])  # Reshape to the original 2D shape
    binary_image[binary_image == 1] = 255  # Set label 1 to white
    binary_image[binary_image == 0] = 0    # Set label 0 to black
    
    return binary_image.astype(np.uint8) 


def superpixel(image, num_superpixels, compactness):
    lab_image = cv.cvtColor(image, cv.COLOR_BGR2Lab)

    # Create SLIC superpixel segmentation object
    slic = cv.ximgproc.createSuperpixelSLIC(
        lab_image, 
        algorithm=cv.ximgproc.SLIC,  # You can use SLICO for adaptive compactness
        region_size=int(np.sqrt(image.shape[0] * image.shape[1] / num_superpixels)),
        ruler=compactness
    )
    
    # Apply the SLIC algorithm
    slic.iterate(10)  # Number of iterations, adjust for finer segmentation
    
    # Get the labels and mask
    labels = slic.getLabels()  # Each pixel gets a superpixel label
    mask = slic.getLabelContourMask()  # Boundary mask for visualization

    # Optional: Visualize superpixels
    segmented_image = image.copy()
    segmented_image[mask == 255] = [0, 0, 255]

    return segmented_image

def separate(image):
    #Get canny
    can = cv.Canny(image,150,150)
    #Classify as foliage or trunk
    nimage = np.zeros_like(image)
    brown = np.array([99, 71, 22])
    green = np.array([53, 150, 53])
    other = np.array([255,0,220])
    thresh = 200
    for x in range(np.shape(image)[1]):
        for y in range(np.shape(image)[0]):
            rgb = image[y][x]
            distanceb = np.linalg.norm(brown - rgb)
            distanceg = np.linalg.norm(green - rgb)
            if can[y][x] == 0:
                if distanceb <= distanceg and distanceb < thresh:
                    nimage[y][x] = brown
                elif distanceg < thresh:
                    nimage[y][x] = green
                else:
                    nimage[y][x] = other
            else: nimage[y][x] == [0,0,0]
    return nimage