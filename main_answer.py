
import numpy as np
import uvicorn
import requests
import json
from pathlib import Path
from urllib.parse import urljoin
from typing import Literal, List, Dict, Tuple
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from src.llm_answer import LLMService
from fastapi import FastAPI, Body, HTTPException

path_config = Path.cwd()/'config'
with open(path_config/'config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

path_model =Path.cwd()/config['folder_model']
path_model .mkdir(exist_ok=True)

app = FastAPI()

WINDOW_SIZE = 10240
model_path = hf_hub_download(**config['model_llm_name'], cache_dir= path_model)
llama = Llama(model_path=model_path, n_ctx=WINDOW_SIZE,verbose=False)

llm_service = LLMService (llama, config)

# Эндпоинт: генерация ответа на вопрос 
@app.post("/answer_question")
async def answer_question(query: str = Body(...)):

    base_url = 'http://localhost:8000/' 
    endpoint_indexer = 'search_texts'
    url = urljoin(base_url, endpoint_indexer)
    # запрос поиска схожих  текстов 
    response = requests.post(
    url, json.dumps(query),
    headers={"Content-Type": "application/json"}
    )
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Ошибка запроса к indexer: {response.text}")

    result = response.json()

    # return {"similarity_scores": similarity_scores.tolist(), "retrieved_texts": retrieved_texts }
    similarity_scores = np.array(result['similarity_scores'])
    retrieved_texts = result['retrieved_texts']

    # prompt_prepare(self, query: str, similarity_scores: np.ndarray, content: List[str]) -> str:
    prompt = llm_service.prompt_prepare(query, similarity_scores, retrieved_texts)
    answer = llm_service.answer_question(prompt)
    return {
        'query': query, 
        'answer': answer,
        'result_indexer': result
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_answer:app", host="0.0.0.0", port=8001)

