# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import tensorflow as tf
import contextlib


print("Converting the model for the Edge TPU ...")

pb_file_directory = 'pidTest.pb'
     
# getting the graph def
def load_graph(frozen_graph_path):
    # Load the frozen graph
    with tf.io.gfile.GFile(frozen_graph_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        
    # Import the graph definition into the default graph
    with tf.compat.v1.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    
    return {'graph_def': graph_def, 'graph': graph}

frozen_graph = load_graph(pb_file_directory)


# flops = tf.compat.v1.profiler.profile(graph=frozen_graph['graph'],\
#      options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
# print(flops.total_float_ops)


# for op in frozen_graph['graph'].get_operations():
#     print(frozen_graph['graph'].get_tensor_by_name(op.name + ":0"))

# Input images
data_yx = np.array([np.load("inputs/numpy_fp32/convertedInput_" + str(i) + ".npy").reshape((2,16,32,1))[0] for i in range(1000)])
data_yz = np.array([np.load("inputs/numpy_fp32/convertedInput_" + str(i) + ".npy").reshape((2,16,32,1))[1] for i in range(1000)])

# Definition of a representative dataset for full integer quantization
# It allows the converter to estimate a dynamic range for all the variable data
def representative_data_gen():
  for yx, yz in zip(tf.data.Dataset.from_tensor_slices(data_yx).batch(1).take(100), tf.data.Dataset.from_tensor_slices(data_yz).batch(1).take(100)):
    yield [yx, yz]


# Converting a GraphDef from file.
converter = tf.compat.v1.lite.TFLiteConverter(
  load_graph(pb_file_directory)['graph_def'], 
  input_tensors=None,
  output_tensors=None,
  input_arrays_with_shape= [('x', [1,16, 32, 1]), ('x_1', [1,16,32,1])],
  output_arrays=['Identity'] 
  )
# input_arrays_with_shape important for edgetpu compiler, cannot take dynamic shape
converter.output_format = tf.compat.v1.lite.constants.TFLITE
converter.representative_dataset = representative_data_gen

# converter.drop_control_dependency = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8




tflite_model = converter.convert()
open("pidTest_quantizedinputs.tflite", "wb").write(tflite_model)






