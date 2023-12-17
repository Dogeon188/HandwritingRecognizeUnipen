import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_digit(img_array, title: str = None):
    plt.imshow(img_array, cmap='gray')
    plt.title(title, fontname="monospace")
    plt.colorbar()
    plt.show()

def plot_digits(img_arrays, labels, predictions=None, shape: tuple[int, int]=(5, 5)):
    plt.figure(figsize=(16, 20))
    for index, (img_array, label) in enumerate(zip(img_arrays, labels)):
        plt.subplot(shape[0], shape[1], index + 1)
        plt.imshow(img_array, cmap='gray')
        if predictions is None:
            plt.title(f"{chr(label + 32)} ({label})")
        else:
            pred = tf.argmax(predictions[index]).numpy()
            plt.title(
                f"'{chr(label + 32)}'({label})→'{chr(pred + 32)}'({pred})",
                color="green" if pred == label else "red", fontname="monospace")
    plt.show()

def plot_filters(filters, shape: tuple[int, int]=(4, 8)):
    plt.figure(figsize=(16, 8))
    for index in range(filters.shape[-1]):
        plt.subplot(shape[0], shape[1], index + 1, )
        plt.imshow(filters[:, :, 0, index], cmap='coolwarm')
        plt.axis('off')
        plt.colorbar()
    plt.show()
