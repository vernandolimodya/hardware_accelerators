import os
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

model = load_model('model.hf5')
sparsity = 0.5

# I will use no validation set -- sorry for the bad practice, because I just want to obtain a sparse model in the end
data_yx = np.array([np.load("inputs/numpy_fp32/convertedInput_" + str(i) + ".npy").reshape((2,16,32,1))[0] for i in range(1000)])
data_yz = np.array([np.load("inputs/numpy_fp32/convertedInput_" + str(i) + ".npy").reshape((2,16,32,1))[1] for i in range(1000)])
labels = np.random.randint(1, 26, size=(1000,))

# clone the model for it to make it sparse
sparse_model = tf.keras.models.clone_model(model)

# sparse the model
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(sparsity, 0),
    'block_size': (1, 1),
    'block_pooling_type': 'AVG'
}

sparse_model = tfmot.sparsity.keras.prune_low_magnitude(sparse_model, **pruning_params)

# "retrain" the model
sparse_model.compile(optimizer = 'adam',
              loss = 'mse',
              metrics = ['sparse_categorical_accuracy']
              )
sparse_model.fit(x=[data_yx, data_yz], y=labels, validation_split=0.2, epochs=1, callbacks=tfmot.sparsity.keras.UpdatePruningStep())



# saving everything #

if not os.path.exists("tpu_models"):
    os.mkdir("tpu_models")

if not os.path.exists("vpu_models"):
    os.mkdir("vpu_models")


# ------------------------ for quantization --------------------------- #

# Definition of a representative dataset for full integer quantization
# It allows the converter to estimate a dynamic range for all the variable data
def representative_data_gen():
  for yx, yz in zip(tf.data.Dataset.from_tensor_slices(data_yx).batch(1).take(100), tf.data.Dataset.from_tensor_slices(data_yz).batch(1).take(100)):
    yield [yx, yz]
# --------------------------------------------------------------------- #

# ------------------------------------------- for the VPU ------------------------------------------ #
mod_vpu_dir_name = "vpu_models/sparsity_" + str(sparsity) + "/"

sparse_model.save(mod_vpu_dir_name + "saved_model/")

# save in frozen graph format #
imported = tf.saved_model.load(mod_vpu_dir_name + "saved_model/")
# retrieve the concrete function and freeze
concrete_func = imported.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
frozen_func = convert_variables_to_constants_v2(concrete_func,
                                            lower_control_flow=False,
                                            aggressive_inlining=True)

# # retrieve GraphDef and save it into .pb format
# graph_def = frozen_func.graph.as_graph_def(add_shapes=True)
# tf.io.write_graph(graph_def, mod_vpu_dir_name, 'frozen_graph.pb', as_text=False)


# ------------------------------------------- for the TPU ------------------------------------------ #
mod_tpu_dir_name = "tpu_models/"

sparse_model = tfmot.sparsity.keras.strip_pruning(sparse_model)

q_converter = tf.lite.TFLiteConverter.from_keras_model(sparse_model)
q_converter.representative_dataset = representative_data_gen
q_converter.optimizations = [tf.lite.Optimize.DEFAULT, tf.lite.Optimize.EXPERIMENTAL_SPARSITY]

# Ensure that if any ops can't be quantized, the converter throws an error
q_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

q_tflite_model = q_converter.convert()
open(mod_tpu_dir_name + "sparsity_" + str(sparsity) + ".tflite", 'wb').write(q_tflite_model)
