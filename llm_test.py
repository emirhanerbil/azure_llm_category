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

sampling_params = SamplingParams(
    temperature=0.2,
    top_p=0.9,
    max_tokens=64,
    stop=["\n"]
    )
prompt = "hello, what is the capital of Turkey?"
outputs = llm.generate([prompt], sampling_params)
response = outputs[0].outputs[0].text.strip()
print(response)