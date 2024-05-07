import os
# from typing import Concatenate
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, MaxPooling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from profile_model import op_stats


input_size = 32
inception_types = ['naive','dimensionality_red']

for type in inception_types:
  input = Input(shape=(input_size,input_size,1), name='input')
  conv_big = Conv2D(filters=128, kernel_size=(1,1), padding='same', name='conv_big')(input)

  if type == 'naive':
    conv_1x1 = Conv2D(filters=32, kernel_size=(1,1), padding='same', name='conv_1x1')(conv_big)
    conv_3x3 = Conv2D(filters=32, kernel_size=(3,3), padding='same', name='conv_3x3')(conv_big)
    conv_5x5 = Conv2D(filters=32, kernel_size=(5,5), padding='same', name='conv_5x5')(conv_big)
    maxpool_3x3 = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='maxpool_3x3')(conv_big)

    concatenate = Concatenate(axis=-1, name='concatenate')([conv_1x1, conv_3x3, conv_5x5, maxpool_3x3])

  elif type == 'dimensionality_red':
    conv_1x1_0 = Conv2D(filters=32, kernel_size=(1,1), padding='same', name='conv_1x1_0')(conv_big)
    conv_1x1_1 = Conv2D(filters=32, kernel_size=(1,1), padding='same', name='conv_1x1_1')(conv_big)
    maxpool_3x3 = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='maxpool_3x3')(conv_big)

    conv_3x3 = Conv2D(filters=32, kernel_size=(3,3), padding='same', name='conv_3x3')(conv_1x1_0)
    conv_5x5 = Conv2D(filters=32, kernel_size=(5,5), padding='same', name='conv_5x5')(conv_1x1_1)
    conv_1x1_2 = Conv2D(filters=32, kernel_size=(1,1), padding='same', name='conv_1x1_2')(maxpool_3x3)
    
    conv_1x1_3 = Conv2D(filters=32, kernel_size=(1,1), padding='same', name='conv_1x1_3')(conv_big)

    concatenate = Concatenate(axis=-1, name='concatenate')([conv_1x1_3, conv_3x3, conv_5x5, conv_1x1_2])

  model = Model(inputs=[input], outputs=[concatenate])

  model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

  # fake_dataset = [np.random.randint(-128, 127, size=(input_size,input_size,1)).astype('float32') for z in range(1000)]
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

  model_name = "inception_" + str(type)
  if not os.path.isdir("tpu_models"):
    os.mkdir("tpu_models")
  with open("tpu_models/" + model_name + ".tflite", 'wb') as f:
    f.write(tflite_model)


  ####### SAVED MODEL + .PB FOR MYRIAD VPU #########

  if not os.path.isdir("vpu_models"):
    os.mkdir("vpu_models")

  if not os.path.isdir("vpu_models/inception_" + str(type) + "/"):
    os.mkdir("vpu_models/inception_" + str(type) + "/")

  model.save("vpu_models/inception_" + str(type) + "/saved_model/")

  # save in frozen graph format #
  imported = tf.saved_model.load("vpu_models/inception_" + str(type) + "/saved_model/")
  # retrieve the concrete function and freeze
  concrete_func = imported.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
  frozen_func = convert_variables_to_constants_v2(concrete_func,
                                                  lower_control_flow=False,
                                                  aggressive_inlining=True)

  # retrieve GraphDef and save it into .pb format
  graph_def = frozen_func.graph.as_graph_def(add_shapes=True)
  tf.io.write_graph(graph_def, "vpu_models/inception_" + str(type) + "/", 'frozen_graph.pb', as_text=False)


  # op_stats # 
  op_stats(model).to_csv("vpu_models/inception_" + type + "/op_stat.csv")


