# optimum-cli export onnx --model BAAI/bge-small-en-v1.5
# --task feature-extraction --optimize O4 --device cuda bge_Opt_04_cu118

from optimum.onnxruntime import (
    ORTOptimizer, 
    OptimizationConfig,
    ORTModelForFeatureExtraction,
)

save_dir = "bge_onnx"

# export to onnx
model = ORTModelForFeatureExtraction.from_pretrained(
    "BAAI/bge-small-en-v1.5", export=True,
)
model.save_pretrained(save_dir)

# optimize graph
optimization_config = optimization_config = OptimizationConfig(
    optimization_level=2,
    optimize_for_gpu=True,
    fp16=True,  # atol: 1e-04
    enable_transformers_specific_optimizations=True,
    disable_layer_norm_fusion=False,  # atol: 1e-05
    disable_attention_fusion=True,  # far off
    disable_skip_layer_norm_fusion=False,  # atol: 1e-05
    disable_bias_skip_layer_norm_fusion=False,  # atol: 1e-05
    disable_bias_gelu_fusion=False,  # atol: 1e-05
    disable_embed_layer_norm_fusion=False,  # atol: 1e-05
    enable_gelu_approximation=True,  # acc off: 1e-03, not compatible with enable_gemm_fast_gelu_fusion
    use_multi_head_attention=True,  # atol: 1e-05
    enable_gemm_fast_gelu_fusion=False,  # atol: 1e-05, not compatible with enable_gelu_approximation
    disable_rotary_embeddings=False,  # atol: 1e-05
)
optimizer = ORTOptimizer.from_pretrained(save_dir)
opt_save_dir = "bge_onnx_opt"
optimizer.optimize(save_dir=opt_save_dir, optimization_config=optimization_config)