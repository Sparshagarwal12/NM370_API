import matplotlib.pyplot as plt
import cv2


def hist(image, name: str):
    _ = plt.hist(image[:, :, 0].ravel(), bins=256, color='red', alpha=0.5)
    _ = plt.hist(image[:, :, 1].ravel(), bins=256, color='Green', alpha=0.5)
    _ = plt.hist(image[:, :, 2].ravel(), bins=256, color='Blue', alpha=0.5)
    _ = plt.xlabel('Intensity Value')
    _ = plt.ylabel('Pixel Count')
    _ = plt.legend(['Red_Channel', 'Green_Channel', 'Blue_Channel'])

    plt.title("Histogram")
    plt.savefig(name)
