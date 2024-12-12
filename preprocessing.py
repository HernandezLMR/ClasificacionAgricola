import cv2 as cv
import numpy as np
import os
import shutil
from clustering import kmeans, superpixel, separate

def preprocess_image(img,resize = (500,500)):
    image = img
    # Apply Gaussian blur to reduce noise
    image = cv.GaussianBlur(image,(3,3),0)
    
    # Resize to standard
    image = cv.resize(img,resize,interpolation=cv.INTER_CUBIC)

    # Convert the image to grayscale
    #image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply canny edge detection
    #image = cv.Canny(image, 100, 200)

    return image

def segment_image(img):
    image = img
    #image = kmeans(image)
    image = separate(image)
    return(image)

if __name__ == '__main__':
    db ="Agricultural-crops"
    segmentTest = "SegmentTest"
    images = []
    shutil.rmtree(segmentTest)
    os.mkdir(segmentTest)

    for d in os.listdir(db):
        folder = os.path.join(db,d)
        for i in os.listdir(folder):
            imPath = os.path.join(folder,i)
            image = cv.imread(imPath)
            image = preprocess_image(image)
            images.append(image)

    for n,image in enumerate(images):
        if n%2 == 0:
            cv.imshow("Original",image)
            image = segment_image(image)
            name = str(n)+".jpg"
            cv.imwrite(os.path.join(segmentTest, name),image)
            cv.imshow("Segmented",image)
            cv.waitKey(0)
            cv.destroyAllWindows()
    import winsound
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 2000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)