# Handwritting Recognition with Simple Neural Network

Just a simple attempt to build a neural network for handwritting recognition using [TensorFlow](https://www.tensorflow.org/).

The repository is a part of a project for the Hardware Design and Lab course. The implementation of the model on FPGA can be found in [Dogeon188/BestIDE](https://github.com/Dogeon188/BestIDE).

The repository also contains a simple GUI for testing the model or generating custom training data.

The dataset used for training is based on [sueiras/handwritting_characters_database](https://github.com/sueiras/handwritting_characters_database), but with our slight additions and modifications.

## Requirements

- Python 3.12
- TensorFlow 2.15.0
- Numpy 1.26.2
- Matplotlib 3.8.2
- PyQt5 5.15.10

## Model

- The model is a simple convolutional neural network (CNN) with **3 convolutional layers** and **2 fully connected layers**.
- The activation function is **ReLU**.
- Input is a 32x32 grayscale image (rescaled from the original 64x64 data size).
- Output is a vector of 96 confidences, index corresponding to the ASCII code of the character minus 32. (`' ' := 0`, `'!' := 1`, `'"' := 2`, etc.)
- Parameters are limited in size and in range of [-2, 2], in order to fit the model into a Basys 3 FPGA board.

![Visualization of the model.](doc/model_vis.png)

## Performance

Metric           | Value
-----------------|----------------------
Parameters       | 57280
Overall loss     | 0.2305
Overall accuracy | 0.9056
