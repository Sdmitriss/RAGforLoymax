import faiss
import numpy as np
import pandas as pd
import  json
import pickle
import gc
import  torch
from pathlib  import Path
from src.custom_logging import Customlogger
from src.preprocessing import DataPreprocessor
from sentence_transformers import SentenceTransformer
from typing import Literal, List, Dict, Tuple

class FaissIndexerService:
    """
    Сервис для управления FAISS-индексом.
    Основные функции:
    - Создание и сохранение FAISS-индекса.
    - Добавление новых текстов в индекс.
    - Поиск похожих текстов по запросу.
    - Удаление  файлов индекса и текстов(для иниициализации индекса).

    Args:
        model (SentenceTransformer): Модель эмбеддинга предложений.
        logger: Экземпляр логгера(экземпляр Customlogger() модуль costom_logging.py).
        path_faiss (Path): Путь к директории для хранения индекса и связанного  файла с текстами.
        path_json_init (Path): Путь к исходному JSON-файлу с текстами.
        config (Dict): Словарь с конфигурационными параметрами.
    """



    def __init__(self,
                 model: SentenceTransformer,
                 logger,
                 path_faiss: Path,
                 path_json_init,
                 config:  Dict
     ):
        

        self.model = model
        self.logger = logger
        self.path_faiss = path_faiss
        self.path_json_init = path_json_init
        self.config = config
        self.dim_emb = model.get_sentence_embedding_dimension()



    def _create_list_texts(self, path_json: Path, add: bool = False):
          """
        Загружает и очищает тексты из JSON-файла, формируя список текстов для индексирования.
        Сохраняет список текстов в pickle-файл( начальный корпус текстов).
        Args:
            path_json (Path): Путь к JSON-файлу с текстами.
            add (bool): Флаг, указывающий, добавочные ли это тексты (True) или начальный корпус (False).
        """

          MIN_WORDS = self.config['min_words']

          #  инициализируем  DataPreprocessor  обработчик текстов(проверки на пропуски,дубликаты, мин длина текста)
          dp = DataPreprocessor(path_json, self.logger, min_words=MIN_WORDS) 

          # создает атрибут .df_clean датафрейм  с очищенным текстом
          dp.clean() 

          # создает атрибут .list_text - список  очищенных текстов
          if add: #
              path_file = self.path_faiss/self.config['file_name_texts_add']
              self.texts_index_add =dp.list_texts()
          else: 
              path_file = self.path_faiss/self.config['file_name_texts']
              self.texts_index =dp.list_texts()# self.list_text
              with open(path_file, 'wb') as f:
                  pickle.dump(self.texts_index, f)
              self.logger.info(f"Создан список текстов и сохранён в {path_file}")  



    def create_index(self):
         """
        Инициализирует FAISS-индекс:
        - Загружает существующий индекс и тексты, если они есть.
        - Иначе создаёт новый индекс из текстов и сохраняет его.

        """

         MIN_WORDS = self.config['min_words']
         self.index = faiss.IndexFlatL2(self.dim_emb)

         path_index = self.path_faiss/self.config['file_name_index']
         path_file = self.path_faiss/self.config['file_name_texts']


         try:
             self.index = faiss.read_index(str(path_index))
             with open(path_file, 'rb') as f:
                 self.texts_index = pickle.load(f)
             assert len( self.texts_index) ==  self.index.ntotal 
             self.logger.info(f'FAISS-индекс dim: {self.index.ntotal}и связанный список текстов успешно загружены из файлов')

         except:
             self.logger.info('Выполняется инициализация FAISS-индекса и списка текстов')
             if path_file.is_file():
                with open(path_file, 'rb') as f:
                     self.texts_index = pickle.load(f)
                self.logger.info(f"Загружен список текстов из{path_file}")
             else:
                  # базовая  инициализация     
                  self._create_list_texts(self.path_json_init) 

             embs =self.model.encode(self.texts_index, batch_size=16, show_progress_bar=True) 
             self.index.add(embs) 
             faiss.write_index(self.index, str(path_index))
             self.logger.info(f'FAISS-индекс dim: {self.index.ntotal} и связанный список текстов созданы  и сохранены')

    def add_index(self, path_json_add: str):
        """
        Добавляет новые тексты из JSON в существующий индекс.
        Обновляет список текстов и сохраняет его.

        Args:
            path_json_add (Path): Путь к JSON-файлу с новыми текстами.

.       """
        path_json_add = Path(path_json_add)

        #  проверка инициализации индекса + инициализация 
        if not hasattr(self, 'index'):
            self.logger.warning(f"Индекс не инициализирован")
            self.create_index()


        path_index = self.path_faiss/self.config['file_name_index']
        path_file = self.path_faiss/self.config['file_name_texts']
        # обоаботает  текст + создаст атрибут self.texts_index_add =dp.list_texts()(список текстов)
        self._create_list_texts( path_json_add, add=True)

        embs =self.model.encode(self.texts_index_add, batch_size=16, show_progress_bar=True)
        self.index.add(embs)
        faiss.write_index(self.index, str(path_index))
        self.texts_index.extend(self.texts_index_add)

        with open(path_file, 'wb') as f:
            pickle.dump(self.texts_index, f)


    def  search_texts (self, text: str) -> Tuple[np.ndarray, List[str]]:
        """"
        Ищет наиболее похожие тексты в FAISS-индексе по входному запросу.

        Args:
            text (str): Запрос пользователя.

        Returns:
            Tuple[np.ndarray, List[str]]:
                - Массив расстояний  до ближайших векторов.
                - Список текстов, соответствующих ближайшим результатам.
        """
        #  проверка инициализации индекса + инициализация 
        if not hasattr(self, 'index'):
            self.logger.warning(f"Индекс не инициализирован")
            self.create_index()


        emb = self.model.encode(text).reshape(1,-1)
        top_k = self.config['top_k_faiss']
        i, d = self. index.search(emb, k=top_k)
        retrieved_texts = [self.texts_index[int(i)] for i  in d.reshape(-1)]
        return i.reshape(-1) ,retrieved_texts
    
    def delete_index_files(self) -> None:
        """
        Удаляет все файлы из директории индекса FAISS.
        Необходимо для  перед иннициализацией индекса новыми тестами!!!
        """
        for file in  self.path_faiss.iterdir():
            if file.is_file():
                file.unlink()
  

















