@echo off
for %%i in (36) do (
	mo --input_model "vpu_models/depth_%%i/frozen_graph.pb" --input_shape [1,32,32,1] --input "input" --model_name ir_model --output_dir "vpu_models/depth_%%i/"
        mo --input_model "vpu_models/depth_%%i/frozen_graph.pb" --input_shape [1,32,32,1] --input "input" --model_name ir_model_compressed --output_dir "vpu_models/depth_%%i/" --compress_to_fp16
)
