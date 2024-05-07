@echo off
for %%i in ("0.75" "0.5" "0.25" "0.0") do (
	mo --input_model "vpu_models/vgg16_sparsity_%%i/frozen_graph.pb" --input_shape [1,32,32,1] --input "input" --model_name ir_model --output_dir "vpu_models/vgg16_sparsity_%%i/"
        mo --input_model "vpu_models/vgg16_sparsity_%%i/frozen_graph.pb" --input_shape [1,32,32,1] --input "input" --model_name ir_model_compressed --output_dir "vpu_models/vgg16_sparsity_%%i/" --compress_to_fp16
)