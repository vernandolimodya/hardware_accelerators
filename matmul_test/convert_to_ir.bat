@echo off
for %%i in (1 2 4 8 16 32 64 128 256 512 1024) do (
    for %%j in (8192 16384 32768) do (
         mo --input_model "vpu_models_ext2/matmul_%%i_%%j/frozen_graph.pb" --input_shape [1,%%i] --input "input" --model_name ir_model --output_dir "vpu_models_ext2/matmul_%%i_%%j/"
         mo --input_model "vpu_models_ext2/matmul_%%i_%%j/frozen_graph.pb" --input_shape [1,%%i] --input "input" --model_name ir_model_compressed --output_dir "vpu_models_ext2/matmul_%%i_%%j/" --compress_to_fp16
    )
)
