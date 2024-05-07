import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from profile_model import op_stats

model = load_model('model.hf5')

no_of_train_samples = 100

# print(model.summary())
# print(model.layers[-7].name, model.layers[-7].trainable) # max_pooling2d_3, False

for layer in model.layers[:-6]:
    layer.trainable = False # Freeze the weights of existing layers

conv2d_dim_red = Conv2D(filters=32, kernel_size=(1,1), padding='same', name='conv2d_dim_red')(model.layers[-7].output)
flatten = Flatten()(conv2d_dim_red)
dense_0 = Dense(units=256)(flatten)
dense_1 = Dense(units=128)(dense_0)
dropout = Dropout(rate=0.2)(dense_1)
output = Dense(26, activation='relu')(dropout)

opt_model = Model(inputs = model.input, outputs = output)
opt_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

x_train_yx = np.array([ np.load("training/numpy_fp32/samples/convertedInput_" + str(i) + ".npy").reshape((2,16,32,1))[0] for i in range(no_of_train_samples) ])
x_train_yz = np.array([ np.load("training/numpy_fp32/samples/convertedInput_" + str(i) + ".npy").reshape((2,16,32,1))[1] for i in range(no_of_train_samples) ])

y_train = np.array([ 0 for i in range(no_of_train_samples) ])

# if the right ones are already available

# y_train = np.array([ np.load("training/numpy_fp32/labels/convertedInput_" + str(i) + ".npy") for i in range(no_of_train_samples) ])

opt_model.fit([x_train_yx, x_train_yz], y_train, epochs=1, batch_size=1, verbose=1, validation_split=0.2)

# ------------------------ for quantization --------------------------- #

# Input images
data_yx = np.array([np.load("inputs/numpy_fp32/convertedInput_" + str(i) + ".npy").reshape((2,16,32,1))[0] for i in range(1000)])
data_yz = np.array([np.load("inputs/numpy_fp32/convertedInput_" + str(i) + ".npy").reshape((2,16,32,1))[1] for i in range(1000)])


# Definition of a representative dataset for full integer quantization
# It allows the converter to estimate a dynamic range for all the variable data
def representative_data_gen():
  for yx, yz in zip(tf.data.Dataset.from_tensor_slices(data_yx).batch(1).take(100), tf.data.Dataset.from_tensor_slices(data_yz).batch(1).take(100)):
    yield [yx, yz]

# --------------------------------------------------------------------- #

if not os.path.exists("tpu_models"):
    os.mkdir("tpu_models")

# save in .hf5 format #
opt_model.save("tpu_models/optimized_model.h5")

# save in saved model format #
opt_model.save("tpu_models/optimized_model/saved_model/")

# save in .tflite format -- unquantized #
converter = tf.lite.TFLiteConverter.from_keras_model(opt_model)
tflite_model = converter.convert()
with open("tpu_models/optimized_model.tflite", 'wb') as f:
    f.write(tflite_model)

# save in .tflite format -- quantized #
q_converter = tf.lite.TFLiteConverter.from_keras_model(opt_model)
q_converter.representative_dataset = representative_data_gen
q_converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Ensure that if any ops can't be quantized, the converter throws an error
q_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

q_tflite_model = q_converter.convert()
open("tpu_models/optimized_model_quantized.tflite", 'wb').write(q_tflite_model)

# save in frozen graph format #
imported = tf.saved_model.load("tpu_models/optimized_model/saved_model/")
# retrieve the concrete function and freeze
concrete_func = imported.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
frozen_func = convert_variables_to_constants_v2(concrete_func,
                                                lower_control_flow=False,
                                                aggressive_inlining=True)

# retrieve GraphDef and save it into .pb format
graph_def = frozen_func.graph.as_graph_def(add_shapes=True)
tf.io.write_graph(graph_def, "vpu_models/optimized_model/", 'frozen_graph.pb', as_text=False)

# op_stats # 
op_stats(opt_model).to_csv("vpu_models/optimized_model/op_stat.csv")

