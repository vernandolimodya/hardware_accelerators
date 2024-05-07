@echo off

mo --input_model vpu_models/optimized_model/frozen_graph.pb --input_shape "[1,16,32,1],[1,16,32,1]" --input "input_yx,input_yz" --model_name ir_model --output_dir vpu_models/optimized_model --layout "NHWC,NHWC"
mo --input_model vpu_models/optimized_model/frozen_graph.pb --input_shape "[1,16,32,1],[1,16,32,1]" --input "input_yx,input_yz" --model_name ir_model_compressed --output_dir vpu_models/optimized_model --layout "NHWC,NHWC" --compress_to_fp16