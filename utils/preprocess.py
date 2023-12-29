import tensorflow as tf
import os

raw_data_path = "data/curated"
dataset_path = "data/unipen.tfrecord"
dataset_no_cap_path = "data/unipen_no_cap.tfrecord"

def path2img(img_path):
    img = tf.image.decode_png(tf.io.read_file(img_path), channels=1)
    # img = tf.reshape(img, [64, 64])
    maxpool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
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

def load_unipen_dataset():
    target_path = dataset_path
    if not os.path.exists(target_path):
        dataset = build_unipen_dataset()
        tf.data.Dataset.save(dataset, target_path)
    data = tf.data.Dataset.load(target_path)
    data = data.map(lambda img, label: (tf.cast(img, tf.float32) / 255.0, label))
    return data

if __name__ == "__main__":
    print("Building dataset...")
    dataset = build_unipen_dataset()
    print("Saving dataset...")
    dataset.save(dataset_path)
    print("Saved dataset to", dataset_path)
