import tensorflow as tf
import os

raw_data_path = "data/curated"
dataset_path = "data/unipen.tfrecord"
dataset_no_cap_path = "data/unipen_no_cap.tfrecord"

def path2img(img_path):
    img = tf.image.decode_png(tf.io.read_file(img_path), channels=1)
    img = tf.reshape(img, [64, 64])
    label_id = tf.strings.split(img_path, os.path.sep)[2]
    label_id = tf.strings.to_number(label_id, out_type=tf.int32)
    return img, label_id

def build_unipen_dataset(no_cap: bool = True):
    dataset = tf.data.Dataset.list_files(os.path.join(raw_data_path, "*/*.png")).map(path2img)
    if no_cap:
        return dataset.filter(lambda img, label: label < ord('A') or label > ord('Z'))
    return dataset

def load_unipen_dataset(no_cap: bool = True):
    target_path = dataset_no_cap_path if no_cap else dataset_path
    if not os.path.exists(target_path):
        dataset = build_unipen_dataset()
        tf.data.Dataset.save(dataset, target_path)
    return tf.data.Dataset.load(target_path)

if __name__ == "__main__":
    load_unipen_dataset()