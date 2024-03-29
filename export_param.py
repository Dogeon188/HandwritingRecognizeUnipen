# import libraries
import keras
from utils.preprocess import *
from config import *

model = keras.models.load_model(f"data/unipen_model.h5")

def toFixedPoint(x):
    ret = int(x * 2**14 + 0.5)
    if ret < 0:
        # 16 bit signed int
        ret = (1 << 16) + ret
    return ret

with open("data/params/conv_weights.coe", "w") as f:
    f.write("; IP:     Block Memory Generator (Dual Port ROM)\n")
    f.write("; Name:   mem_conv_w\n")
    f.write("; WidthA: 144\n")
    f.write("; DepthA: 2576\n")
    f.write("; WidthB: 144\n")
    f.write("\n")
    f.write("memory_initialization_radix = 16;\n")
    f.write("memory_initialization_vector = \n")
    for layer in filter(lambda x: x.name.startswith("conv"), model.layers):
        weights = layer.get_weights()[0]
        channels = weights.shape[2]
        outputs = weights.shape[3]
        f.write(f"; {layer.name} {weights.shape}\n")
        for j in range(outputs):
            f.write(f";   output {j}\n")
            for i in range(channels):
                for w in map(toFixedPoint, weights[:,:,i,j].flatten()[-1::-1]):
                    f.write(f"{w:04X}")
                f.write(",\n")
    f.write(";\n")

with open("data/params/conv_biases.coe", "w") as f:
    f.write("; IP:    Distributed Memory Generator (ROM)\n")
    f.write("; Name:  mem_conv_b\n")
    f.write("; Depth: 112\n")
    f.write("; Width: 16\n")
    f.write("\n")
    f.write("memory_initialization_radix = 16;\n")
    f.write("memory_initialization_vector = \n")
    for layer in filter(lambda x: x.name.startswith("conv"), model.layers):
        biases = layer.get_weights()[1]
        f.write(f"; {layer.name} {biases.shape}\n")
        for b in map(toFixedPoint, biases.flatten()):
            f.write(f"{b:04X},\n")
    f.write(";\n")

with open("data/params/dense_weights.coe", "w") as f:
    f.write("; IP:     Block Memory Generator (Dual Port ROM)\n")
    f.write("; Name:   mem_dense_w\n")
    f.write("; WidthA: 16\n")
    f.write("; DepthA: 33792\n")
    f.write("; WidthB: 16\n")
    f.write("\n")
    f.write("memory_initialization_radix = 16;\n")
    f.write("memory_initialization_vector = \n")
    for layer in filter(lambda x: x.name.startswith("dense"), model.layers):
        weights = layer.get_weights()[0]
        f.write(f"; {layer.name} {weights.shape}\n")
        for o in range(weights.shape[1]):
            f.write(f";   output {o}\n")
            for w in list(map(toFixedPoint, weights[:,o])):
                f.write(f"{w:04X},\n")
    f.write(";\n")

with open("data/params/dense_biases.coe", "w") as f:
    f.write("; IP:    Distributed Memory Generator (ROM)\n")
    f.write("; Name:  mem_dense_b\n")
    f.write("; Depth: 192\n")
    f.write("; Width: 16\n")
    f.write("\n")
    f.write("memory_initialization_radix = 16;\n")
    f.write("memory_initialization_vector = \n")
    for layer in filter(lambda x: x.name.startswith("dense"), model.layers):
        biases = layer.get_weights()[1]
        f.write(f"; {layer.name} {biases.shape}\n")
        for b in map(toFixedPoint, biases.flatten()):
            f.write(f"{b:04X},\n")
    f.write(";\n")
