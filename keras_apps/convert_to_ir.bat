@echo off
for %%i in ("DenseNet121" "MobileNet" "MobileNetV2" "ResNet50" "ResNet50V2" "VGG16" "VGG19") do (
	mo --input_model "vpu_models/%%i/frozen_graph.pb" --input_shape [1,32,32,1] --input "input" --model_name ir_model --output_dir "vpu_models/%%i/"
        mo --input_model "vpu_models/%%i/frozen_graph.pb" --input_shape [1,32,32,1] --input "input" --model_name ir_model_compressed --output_dir "vpu_models/%%i/" --compress_to_fp16
)