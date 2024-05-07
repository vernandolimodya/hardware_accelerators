import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from profile_model import op_stats


input_size = 32
depths = [i+1 for i in range(36)]

for depth in depths:

  model = Sequential()

  model.add(Input(shape=(input_size,input_size,1), name='input'))

  model.add(Flatten())
  
  for _ in range(depth):
    model.add(Dense(units=1024, activation='relu'))

  model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

  fake_dataset = [np.load("inputs/numpy_fp32_concat/convertedInput_" + str(i) + ".npy").reshape((input_size,input_size,1)).astype('float32') for i in range(1000)]

  def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(fake_dataset).batch(1).take(100):
      yield [input_value]

  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.representative_dataset = representative_data_gen
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  # converter.inference_input_type = tf.int8
  # converter.inference_output_type = tf.int8
  tflite_model = converter.convert()

  model_name = "depth_" + str(depth)
  if not os.path.isdir("tpu_models"):
    os.mkdir("tpu_models")
  with open("tpu_models/" + model_name + ".tflite", 'wb') as f:
    f.write(tflite_model)


  ####### SAVED MODEL + .PB FOR MYRIAD VPU #########

  if not os.path.isdir("vpu_models"):
    os.mkdir("vpu_models")

  if not os.path.isdir("vpu_models/depth_" + str(depth) + "/"):
    os.mkdir("vpu_models/depth_" + str(depth) + "/")

  model.save("vpu_models/depth_" + str(depth) + "/saved_model/")

  # save in frozen graph format #
  imported = tf.saved_model.load("vpu_models/depth_" + str(depth) + "/saved_model/")
  # retrieve the concrete function and freeze
  concrete_func = imported.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
  frozen_func = convert_variables_to_constants_v2(concrete_func,
                                                  lower_control_flow=False,
                                                  aggressive_inlining=True)

  # retrieve GraphDef and save it into .pb format
  graph_def = frozen_func.graph.as_graph_def(add_shapes=True)
  tf.io.write_graph(graph_def, "vpu_models/depth_" + str(depth) + "/", 'frozen_graph.pb', as_text=False)

  op_stats(model).to_csv("vpu_models/depth_" + str(depth) + "/op_stat.csv")
