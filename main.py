from fastapi import FastAPI
import json
from vllm import LLM, SamplingParams

app = FastAPI()

# Global model nesnesi
llm = None

async def load_model():
    global llm
    if llm is None:
        model_name = "Qwen/Qwen2-7B-Instruct"
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=1,
            dtype="float16",
            gpu_memory_utilization=0.8,
            max_model_len=5608
        )

@app.on_event("startup")
async def startup_event():
    await load_model()




def format_chat_prompt(query, cat_list):
    # Qwen2 formatında sistem ve kullanıcı mesajlarını birleştirme
    prompt = (
    "System: Your goal is to identify the most relevant category for the given query. These queries are made to an e-commerce site. "
    "Please select only one product category that is most relevant to the query. "
    "Write only the category name, do not add any extra information.\n\n"
    f"Human: Query: '{query}'. Please choose the category that best matches the query "
    f"from the following categories. If the similarity is low, respond with 'irrelevant'. "
    "The answer should only contain one category.\n\n"
    f"System: Select from these Categories: {cat_list}\n\n"
    "Assistant:")

    return prompt

with open("title_cat_waw.json", "r", encoding="utf-8") as file:
    data = json.load(file)


@app.get("/")
async def home():
    return {"msg": "Hello"}


@app.get("/q/")
async def llm_response(q: str,cid: str):
    global llm
    try:
        cat_list = data[cid]
        prompt = format_chat_prompt(q, cat_list)

        sampling_params = SamplingParams(
            temperature=0.2,
            top_p=0.9,
            max_tokens=64,
            stop=["\n"]
            )
        prompt = format_chat_prompt(q, cat_list)
        outputs = llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text.strip()
        return {"status": "success", "data": q,"cid":cid,"res":response}  
    except Exception as e:
        return {"status": "error","error_msg":str(e), "res": ""}
