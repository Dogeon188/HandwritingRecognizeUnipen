import tensorflow as tf
import os

raw_data_path = "data/curated"
dataset_path = "data/unipen.tfrecord"

def path2img(img_path):
    img = tf.image.decode_png(tf.io.read_file(img_path), channels=1)
    img = tf.reshape(img, [64, 64])
    label_id = tf.strings.split(img_path, os.path.sep)[2]
    label_id = tf.strings.to_number(label_id, out_type=tf.int32)
    return img, label_id

def build_unipen_dataset():
    return tf.data.Dataset.list_files(os.path.join(raw_data_path, "*/*.png")).map(path2img)

def load_unipen_dataset():
    if not os.path.exists(dataset_path):
        dataset = build_unipen_dataset()
        tf.data.Dataset.save(dataset, dataset_path)
        return dataset
    return tf.data.Dataset.load(dataset_path)
