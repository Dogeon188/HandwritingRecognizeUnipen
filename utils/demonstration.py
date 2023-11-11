import numpy as np
import matplotlib.pyplot as plt

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