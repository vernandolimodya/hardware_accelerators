import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


# input_shapes = [int(2**i) for i in np.arange(0, 18)]
# output_shapes = input_shapes

# input_shapes = [int(2**i) for i in np.arange(11,16)]

# input_shapes = [int(2**i) for i in np.arange(15,16)]
# output_shapes = [int(2**i) for i in np.arange(14,16)]

input_shapes = [int(2**15)]
output_shapes = [int(2**15)]


# input_shapes = [1024, 32768, 131072, 65536, 53760, 26880, 107520, 66560, 33280, 16896, 8448, 5120, 256, 128, 26]
# output_shapes = [26]

# for input_shape in input_shapes:
#   for output_shape in output_shapes:
    
#     if not (input_shape < 2**14 and output_shape < 2**14):
#       print(np.log2(input_shape), np.log2(output_shape))


for input_shape in input_shapes:
  for output_shape in output_shapes:

    input = Input(shape=(input_shape,), name='input')

    dense = Dense(output_shape, name='output_1', trainable=False, use_bias=False)(input)

    model = Model(inputs=[input], outputs=[dense])

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    fake_dataset = [np.random.randint(-128, 127, size=(input_shape,)).astype('float32') for z in range(1000)]
    def representative_data_gen():
      for input_value in tf.data.Dataset.from_tensor_slices(fake_dataset).batch(1).take(100):
        yield [input_value]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    model_name = "matmul_" + str(input_shape) + "_" + str(output_shape)
    if not os.path.isdir("tpu_models_ext3"):
      os.mkdir("tpu_models_ext3")
    with open("tpu_models_ext3/" + model_name + ".tflite", 'wb') as f:
      f.write(tflite_model)


    ####### SAVED MODEL + .PB FOR MYRIAD VPU #########

    if not os.path.isdir("vpu_models_ext3"):
      os.mkdir("vpu_models_ext3")

    if not os.path.isdir("vpu_models_ext3/matmul_" + str(input_shape) + "_" + str(output_shape) + "/"):
      os.mkdir("vpu_models_ext3/matmul_" + str(input_shape) + "_" + str(output_shape) + "/")

    model.save("vpu_models_ext3/matmul_" + str(input_shape) + "_" + str(output_shape) + "/saved_model/")

    # save in frozen graph format #
    imported = tf.saved_model.load("vpu_models_ext3/matmul_" + str(input_shape) + "_" + str(output_shape) + "/saved_model/")
    # retrieve the concrete function and freeze
    concrete_func = imported.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    frozen_func = convert_variables_to_constants_v2(concrete_func,
                                                    lower_control_flow=False,
                                                    aggressive_inlining=True)

    # retrieve GraphDef and save it into .pb format
    graph_def = frozen_func.graph.as_graph_def(add_shapes=True)
    tf.io.write_graph(graph_def, "vpu_models_ext3/matmul_" + str(input_shape) + "_" + str(output_shape) + "/", 'frozen_graph.pb', as_text=False)


