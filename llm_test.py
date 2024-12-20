from vllm import LLM, SamplingParams

# Modeli yükle
model_name="Qwen/Qwen2-7B-Instruct"
# Initialize the model with vLLM
llm = LLM(
    model=model_name,
    trust_remote_code=True,  # Required for Qwen models
    tensor_parallel_size=1,   # Adjust based on available GPUs
    dtype="float16",         # Using float16 for better memory efficiency
    gpu_memory_utilization=0.8,
    max_model_len=5608
)

print("done")