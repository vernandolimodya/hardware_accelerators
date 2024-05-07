@echo off
for %%i in (3 5 7 9 11 13 15 17 19 21 23) do (
	mo --input_model "vpu_models/conv2d_%%i/frozen_graph.pb" --input_shape [1,32,32,1] --input "input" --model_name ir_model --output_dir "vpu_models/conv2d_%%i/"
        mo --input_model "vpu_models/conv2d_%%i/frozen_graph.pb" --input_shape [1,32,32,1] --input "input" --model_name ir_model_compressed --output_dir "vpu_models/conv2d_%%i/" --compress_to_fp16
)
