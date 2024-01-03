import tensorflow as tf
import os
import keras
import numpy as np

raw_data_path = "data/curated"
dataset_path = "data/unipen.tfrecord"
dataset_no_cap_path = "data/unipen_no_cap.tfrecord"

def path2img(img_path):
    img = tf.image.decode_png(tf.io.read_file(img_path), channels=1)
    # img = tf.reshape(img, [64, 64])
    maxpool = tf.keras.layers.MaxPooling2D((2, 2))
    img = maxpool(tf.reshape(img, [1, 64, 64, 1]))
    img = tf.reshape(img, [32, 32])
    label_id = tf.strings.split(img_path, os.path.sep)[2]
    label_id = tf.strings.to_number(label_id, out_type=tf.int32)
    return img, label_id

def build_unipen_dataset():
    """
    data biased by 32
    label range: [32, 126] - 32 = [0, 94]
    """
    dataset = tf.data.Dataset.list_files(os.path.join(raw_data_path, "*/*.png")).map(path2img)
    dataset = dataset.map(lambda img, label: (img, label - 32))
    return dataset

augmentLayers = [
    keras.layers.RandomRotation(0.02, fill_mode='constant'),
    keras.layers.RandomTranslation(0.05, 0.05, fill_mode='constant'),
    keras.layers.RandomZoom((0, 0.05), fill_mode='constant'),
]

def augment_func(data, label):
    data = tf.expand_dims(data, axis=-1)
    for layer in augmentLayers:
        data = layer(data)
    data = tf.squeeze(data, axis=-1)
    return data, label

def load_unipen_dataset(augment=True, repeat=4):
    if not os.path.exists(dataset_path):
        dataset = build_unipen_dataset()
        tf.data.Dataset.save(dataset, dataset_path)
    dataset = tf.data.Dataset.load(dataset_path)

    dataset = dataset.map(lambda img, label: (tf.cast(img, tf.float32) / 255.0, label))

    if augment:
        dataset = dataset.repeat(repeat).map(augment_func)
    
    return dataset

if __name__ == "__main__":
    print("Building dataset...")
    dataset = build_unipen_dataset()
    print("Saving dataset...")
    dataset.save(dataset_path)
    print("Saved dataset to", dataset_path)
