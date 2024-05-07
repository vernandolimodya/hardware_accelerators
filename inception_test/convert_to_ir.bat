@echo off

mo --input_model "vpu_models/inception_naive/frozen_graph.pb" --input_shape [1,32,32,1] --input "input" --model_name ir_model --output_dir "vpu_models/inception_naive/"
mo --input_model "vpu_models/inception_naive/frozen_graph.pb" --input_shape [1,32,32,1] --input "input" --model_name ir_model_compressed --output_dir "vpu_models/inception_naive/" --compress_to_fp16

mo --input_model "vpu_models/inception_dimensionality_red/frozen_graph.pb" --input_shape [1,32,32,1] --input "input" --model_name ir_model --output_dir "vpu_models/inception_dimensionality_red/"
mo --input_model "vpu_models/inception_dimensionality_red/frozen_graph.pb" --input_shape [1,32,32,1] --input "input" --model_name ir_model_compressed --output_dir "vpu_models/inception_dimensionality_red/" --compress_to_fp16