
import pandas as pd
import  json
import yaml
import pickle
import gc
import  torch
import numpy as np
import uvicorn
from pathlib  import Path
from src.custom_logging import Customlogger
from src.preprocessing import DataPreprocessor
from src.faiss_service import FaissIndexerService
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
from typing import Literal, List, Dict, Tuple
from fastapi import FastAPI, Body
from fastapi import HTTPException

# Загружаем  словарь  config
path_config = Path.cwd()/'config'
with open(path_config/'config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

app = FastAPI()
# Папка  для
path_data = Path.cwd()/'data_raw'
path_data.mkdir(exist_ok=True)

# Папка  для хранения моделей
path_model =Path.cwd()/config['folder_model']
path_model .mkdir(exist_ok=True)

# Папка для записи  файлов индекса и связанных текстов
path_faiss =  Path.cwd()/config['folder_model']/'faiss'
path_faiss.mkdir(exist_ok=True)




# # Путь к исходному JSON-файлу для инициализации Faiss, имя берётся из конфигурации
path_json_init = Path.cwd()/'data_raw'/config['name_json_init']

logger_1 = Customlogger()

#  Инициализация модели для получения эмбеддингов
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_emb= SentenceTransformer(config['model_embed_name'], cache_folder=path_model)
model_emb.max_seq_length = 512
model_emb.to(device)

#инициализация экземпляра класса FaissIndexerService
indexer = FaissIndexerService(model_emb, logger_1, path_faiss, path_json_init, config)



#  Эндпоинт  инициализации храанилища
@app.post("/create_index")
def create_index():
    indexer.create_index()
    return {"status": "Index created successfully"}

#  Эндпоинт для добавления текстов в индекс из JSON-файла по указанному пути
@app.post("/add_index")
def add_index(path_json_add: str = Body(...)):
    """

    """
    path = Path(path_json_add)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found at the specified path")
    indexer.add_index(path_json_add)
    return {"status": f"Added texts from file {path_json_add} to index"}

# Эндпоинт для поиска схожих текстов.
# Принимает строку (json строка) и возвращает список расстояний и найденных текстов
@app.post("/search_texts")
async def search_texts(text: str = Body(...)):

    try:
        similarity_scores, retrieved_texts = indexer.search_texts(text)
    except Exception:
        raise HTTPException(status_code=500, detail="Vector database search error")   

    return {"similarity_scores": similarity_scores.tolist(), "retrieved_texts": retrieved_texts }

# Удаляет файлы сотояния хранилища, после возможна инициализация на новых данных
@app.delete("/delete_index_files")
def delete_index_files():
    indexer.delete_index_files()
    return {"status": "Index files deleted"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_indexer:app", host="0.0.0.0", port=8000)

