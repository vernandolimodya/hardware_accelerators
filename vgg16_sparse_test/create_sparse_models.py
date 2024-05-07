"""
Model needs to be re-trained in order to be pruned by TFMOT
Look at:
https://stackoverflow.com/questions/64686187/is-re-training-required-on-a-model-pruned-using-tfmot

How pruning works in tensorflow:
https://blog.tensorflow.org/2021/07/build-fast-sparse-on-device-models-with-tf-mot-pruning-api.html


Pruning in Keras:
https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras

LOOK AT THIS:
https://www.tensorflow.org/model_optimization/guide/pruning/comprehensive_guide 


LOOK AT THIS (2):
https://stackoverflow.com/questions/67613483/saved-model-file-size-is-the-same-after-pruning-with-tensorflow-model-optimizati


VISUALIZING WEIGHTS:
https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_sparsity_2_by_4#visualize_and_check_weights

"""

import os
import tempfile
import zipfile
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras import Model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from profile_model import op_stats

input_size = 32
classes = 26
pooling = 'max'
classifier_activation = 'relu'
sparsities = [0.75, 0.50, 0.25, 0.00]

benchmark_model = tf.keras.applications.VGG16(include_top=False, input_shape=(input_size,input_size,3), classes=classes, pooling=pooling, classifier_activation=classifier_activation)

greyscale_input = tf.keras.layers.Input(shape=(input_size,input_size,1), name='input')
conv2d_beginning = Conv2D(filters=3, kernel_size=(1,1), name='conv2d_beginning')(greyscale_input)
benchmark_output = benchmark_model(conv2d_beginning)
pid_output = Dense(26)(benchmark_output)

model = Model(inputs=[greyscale_input], outputs=[pid_output])

# ------------------------------------------------------------------------------------------- #

# I will use no validation set -- sorry for the bad practice, because I just want to obtain a sparse model in the end
data_yx = np.array([ np.load("training_inputs/numpy_fp32/samples/convertedInput_" + str(i) + ".npy").reshape((2,16,32,1))[0] for i in range(1000)])
data_yz = np.array([ np.load("training_inputs/numpy_fp32/samples/convertedInput_" + str(i) + ".npy").reshape((2,16,32,1))[1] for i in range(1000)])
print("Concatenating Training Data ... ")
data_yxyz = np.array([ np.concatenate((data_yz[i], data_yx[i])) for i in range(1000)])

labels = np.array([ int(np.load("training_inputs/numpy_fp32/labels/convertedInput_" + str(i) + ".npy")) for i in range(1000)])


def get_gzipped_model_size(file):
    # Returns size of gzipped model, in bytes.

    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)

    return os.path.getsize(zipped_file)

for sparsity in sparsities:
    # clone the model for it to make it sparse
    sparse_model = tf.keras.models.clone_model(model)

    # sparse the model
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(sparsity, 0)
    }

    sparse_model = tfmot.sparsity.keras.prune_low_magnitude(sparse_model, **pruning_params)

    # "retrain" the model
    sparse_model.compile(optimizer = 'adam',
                loss = 'mse',
                metrics = ['sparse_categorical_accuracy']
                )
    sparse_model.fit(x=data_yxyz, y=labels, validation_split=0.2, epochs=5, callbacks=[tfmot.sparsity.keras.UpdatePruningStep()])

    # experiment with pruning size
    model_for_export = tfmot.sparsity.keras.strip_pruning(sparse_model)

    # saving everything #

    if not os.path.exists("tpu_models"):
        os.mkdir("tpu_models")

    if not os.path.exists("vpu_models"):
        os.mkdir("vpu_models")



    # ------------------------ for quantization --------------------------- #

    # Definition of a representative dataset for full integer quantization
    # It allows the converter to estimate a dynamic range for all the variable data
        
    fake_dataset = [np.load("inputs/numpy_fp32_concat/convertedInput_" + str(i) + ".npy").reshape((input_size,input_size,1)).astype('float32') for i in range(1000)]
    def representative_data_gen():
       for input_value in tf.data.Dataset.from_tensor_slices(fake_dataset).batch(1).take(100):
           yield [input_value]
    # --------------------------------------------------------------------- #

    # ------------------------------------------- for the VPU ------------------------------------------ #
    mod_vpu_dir_name = "vpu_models/vgg16_sparsity_" + str(sparsity) + "/"

    tf.keras.models.save_model(model=model_for_export, filepath=mod_vpu_dir_name + "saved_model/", save_format="tf", include_optimizer=False)

    # save in frozen graph format #
    imported = tf.saved_model.load(mod_vpu_dir_name + "saved_model/")
    # retrieve the concrete function and freeze
    concrete_func = imported.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    frozen_func = convert_variables_to_constants_v2(concrete_func,
                                                lower_control_flow=False,
                                                aggressive_inlining=True)

    # # retrieve GraphDef and save it into .pb format
    graph_def = frozen_func.graph.as_graph_def(add_shapes=True)
    tf.io.write_graph(graph_def, mod_vpu_dir_name, 'frozen_graph.pb', as_text=False)


    # ------------------------------------------- for the TPU ------------------------------------------ #
    mod_tpu_dir_name = "tpu_models/"

    q_converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    q_converter.representative_dataset = representative_data_gen
    q_converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Ensure that if any ops can't be quantized, the converter throws an error
    q_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    q_tflite_model = q_converter.convert()

    q_tflite_file = open(mod_tpu_dir_name + "vgg16_sparsity_" + str(sparsity) + ".tflite", 'wb')
    q_tflite_file.write(q_tflite_model)

    # SHOW THE EFFECTS OF PRUNING # 
    print("Zipped file size of vgg16_sparsity_" + str(sparsity) + " : " + str(get_gzipped_model_size(mod_tpu_dir_name + "vgg16_sparsity_" + str(sparsity) + ".tflite")) + " bytes")

