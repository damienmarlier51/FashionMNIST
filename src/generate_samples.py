from tensorflow import keras
from pathlib import Path
import imageio
import os


if __name__ == "__main__":

    file_path = os.path.abspath(__file__)
    project_dir = Path(file_path).resolve().parents[1]

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    iter = 0
    for test_image in test_images:
        iter += 1
        if iter > 10:
            break
        imageio.imsave("{}/samples/{}.png".format(project_dir, iter), test_image)
