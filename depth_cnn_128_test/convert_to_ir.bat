@echo off
for %%i in (1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24) do (
	mo --input_model "vpu_models/depth_conv2d_%%i/frozen_graph.pb" --input_shape [1,32,32,1] --input "input" --model_name ir_model --output_dir "vpu_models/depth_conv2d_%%i/"
        mo --input_model "vpu_models/depth_conv2d_%%i/frozen_graph.pb" --input_shape [1,32,32,1] --input "input" --model_name ir_model_compressed --output_dir "vpu_models/depth_conv2d_%%i/" --compress_to_fp16
)
