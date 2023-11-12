import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

gray_ascii = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "

def print_digit(img_array):
    shape_im = np.shape(img_array)
    for i in range(shape_im[0]):
        for j in range(shape_im[1]):
            gray_scale = img_array[i][j]
            gray_scale = (len(gray_ascii) - 1) - int(gray_scale * (len(gray_ascii) - 1) / 255)
            print(gray_ascii[gray_scale], end='')
        print('')

def plot_digit(img_array, title: str = None):
    plt.imshow(img_array, cmap='gray')
    plt.title(title)
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
                f"{chr(label + 32)} ({label}) -> {chr(pred + 32)} ({pred})",
                color="green" if pred == label else "red")
    plt.show()

def plot_filters(filters, shape: tuple[int, int]=(4, 8)):
    plt.figure(figsize=(16, 8))
    for index in range(filters.shape[-1]):
        plt.subplot(shape[0], shape[1], index + 1, )
        plt.imshow(filters[:, :, 0, index], cmap='coolwarm')
        plt.axis('off')
        plt.colorbar()
    plt.show()
