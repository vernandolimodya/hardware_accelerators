import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras import Model
from profile_model import op_stats


input_size = 32
classes = 26
pooling = 'max'
classifier_activation = 'relu'


models_dict = {'VGG16': tf.keras.applications.VGG16(include_top=False, input_shape=(input_size,input_size,3),
                                                           classes=classes, pooling=pooling, 
                                                           classifier_activation=classifier_activation),
              'VGG19': tf.keras.applications.VGG19(include_top=False, input_shape=(input_size,input_size,3),
                                                           classes=classes, pooling=pooling, 
                                                           classifier_activation=classifier_activation),
              'ResNet50': tf.keras.applications.ResNet50(include_top=False, input_shape=(input_size,input_size,3),
                                                           classes=classes, pooling=pooling, 
                                                           classifier_activation=classifier_activation),
              'ResNet50V2': tf.keras.applications.ResNet50V2(include_top=False, input_shape=(input_size,input_size,3),
                                                           classes=classes, pooling=pooling, 
                                                           classifier_activation=classifier_activation),
              'MobileNet': tf.keras.applications.MobileNet(include_top=False, input_shape=(input_size,input_size,3),
                                                           classes=classes, pooling=pooling, 
                                                           classifier_activation=classifier_activation),
              #'MobileNetV2': tf.keras.applications.MobileNetV2(include_top=False, input_shape=(input_size,input_size,3),
              #                                             classes=classes, pooling=pooling, 
              #                                             classifier_activation=classifier_activation),
              'DenseNet121': tf.keras.applications.DenseNet121(include_top=False, input_shape=(input_size,input_size,3),
                                                           classes=classes, pooling=pooling, 
                                                           classifier_activation=classifier_activation)                                           
              }








for model_name, benchmark_model in models_dict.items():

  print("Creating AFIS version of " + model_name + " ...")

  greyscale_input = tf.keras.layers.Input(shape=(input_size,input_size,1), name='input')
  conv2d_beginning = Conv2D(filters=3, kernel_size=(1,1), name='conv2d_beginning')(greyscale_input)
  benchmark_output = benchmark_model(conv2d_beginning)
  pid_output = Dense(26)(benchmark_output)

  model = Model(inputs=[greyscale_input], outputs=[pid_output])

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

  model_name = model_name
  if not os.path.isdir("tpu_models"):
    os.mkdir("tpu_models")
  with open("tpu_models/" + model_name + ".tflite", 'wb') as f:
    f.write(tflite_model)


  ####### SAVED MODEL + .PB FOR MYRIAD VPU #########

  if not os.path.isdir("vpu_models"):
    os.mkdir("vpu_models")

  if not os.path.isdir("vpu_models/" + model_name + "/"):
    os.mkdir("vpu_models/" + model_name + "/")

  model.save("vpu_models/" + model_name + "/saved_model/")

  # save in frozen graph format #
  imported = tf.saved_model.load("vpu_models/" + model_name + "/saved_model/")
  # retrieve the concrete function and freeze
  concrete_func = imported.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
  frozen_func = convert_variables_to_constants_v2(concrete_func,
                                                  lower_control_flow=False,
                                                  aggressive_inlining=True)

  # retrieve GraphDef and save it into .pb format
  graph_def = frozen_func.graph.as_graph_def(add_shapes=True)
  tf.io.write_graph(graph_def, "vpu_models/" + model_name + "/", 'frozen_graph.pb', as_text=False)

  op_stats(model).to_csv("vpu_models/" + model_name + "/op_stat.csv")

