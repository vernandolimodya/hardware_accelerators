@echo off
for %%i in (2 4 6 8 10 12 14 16) do (
	mo --input_model "vpu_models/maxpooling2d_%%i/frozen_graph.pb" --input_shape [1,32,32,1] --input "input" --model_name ir_model --output_dir "vpu_models/maxpooling2d_%%i/"
        mo --input_model "vpu_models/maxpooling2d_%%i/frozen_graph.pb" --input_shape [1,32,32,1] --input "input" --model_name ir_model_compressed --output_dir "vpu_models/maxpooling2d_%%i/" --compress_to_fp16

	mo --input_model "vpu_models/avgpooling2d_%%i/frozen_graph.pb" --input_shape [1,32,32,1] --input "input" --model_name ir_model --output_dir "vpu_models/avgpooling2d_%%i/"
        mo --input_model "vpu_models/avgpooling2d_%%i/frozen_graph.pb" --input_shape [1,32,32,1] --input "input" --model_name ir_model_compressed --output_dir "vpu_models/avgpooling2d_%%i/" --compress_to_fp16
)
