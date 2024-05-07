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
from tensorflow.keras.models import load_model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

model = load_model('model.hf5')
sparsities = [0.75, 0.50, 0.25, 0.00]

# I will use no validation set -- sorry for the bad practice, because I just want to obtain a sparse model in the end
data_yx = np.array([ np.load("inputs/numpy_fp32/samples/convertedInput_" + str(i) + ".npy").reshape((2,16,32,1))[0] for i in range(1000)])
data_yz = np.array([ np.load("inputs/numpy_fp32/samples/convertedInput_" + str(i) + ".npy").reshape((2,16,32,1))[1] for i in range(1000)])
labels = np.array([ int(np.load("inputs/numpy_fp32/labels/convertedInput_" + str(i) + ".npy")) for i in range(1000)])


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
        # 'block_size': (1, 1),
        # 'block_pooling_type': 'AVG'
    }

    sparse_model = tfmot.sparsity.keras.prune_low_magnitude(sparse_model, **pruning_params)

    # "retrain" the model
    sparse_model.compile(optimizer = 'adam',
                loss = 'mse',
                metrics = ['sparse_categorical_accuracy']
                )
    sparse_model.fit(x=[data_yx, data_yz], y=labels, validation_split=0.2, epochs=10, callbacks=[tfmot.sparsity.keras.UpdatePruningStep()])

    # export this model
    # model_for_export = tfmot.sparsity.keras.strip_pruning(sparse_model)

    
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
    def representative_data_gen():
        for yx, yz in zip(tf.data.Dataset.from_tensor_slices(data_yx).batch(1).take(100), tf.data.Dataset.from_tensor_slices(data_yz).batch(1).take(100)):
            yield [yx, yz]
    # --------------------------------------------------------------------- #

    # ------------------------------------------- for the VPU ------------------------------------------ #
    mod_vpu_dir_name = "vpu_models/pid_sparsity_" + str(sparsity) + "/"

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

    q_tflite_file = open(mod_tpu_dir_name + "pid_sparsity_" + str(sparsity) + ".tflite", 'wb')
    q_tflite_file.write(q_tflite_model)

    # SHOW THE EFFECTS OF PRUNING # 
    print("Zipped file size of pid_sparsity_" + str(sparsity) + " : " + str(get_gzipped_model_size(mod_tpu_dir_name + "pid_sparsity_" + str(sparsity) + ".tflite")) + " bytes")

