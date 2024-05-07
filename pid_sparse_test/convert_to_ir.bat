@echo off
for %%i in ("0.75" "0.5" "0.25" "0.0") do (
	mo --input_model "vpu_models/pid_sparsity_%%i/frozen_graph.pb" --input_shape [1,16,32,1],[1,16,32,1] --input "input_yx,input_yz" --model_name ir_model --output_dir "vpu_models/pid_sparsity_%%i/" --layout "NHWC,NHWC"
        mo --input_model "vpu_models/pid_sparsity_%%i/frozen_graph.pb" --input_shape [1,16,32,1],[1,16,32,1] --input "input_yx,input_yz" --model_name ir_model_compressed --output_dir "vpu_models/pid_sparsity_%%i/" --layout "NHWC,NHWC" --compress_to_fp16
)