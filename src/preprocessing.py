from pathlib import Path
import pandas as pd
import re
import json
import pickle
from typing import Literal, List
from src.custom_logging import Customlogger


class DataPreprocessor:
    '''
    Аргументы:
    - path — путь к JSON-файлу с данными.
    - logger — инициализированный экземпляр логгера из модуля Customlogger.
    - min_words — минимальное количество слов в тексте для векторизации.

    Private методы:
    - _load_json — загружает данные из JSON-файла по заданному пути и возвращает их в виде словаря.
    - _check_missing — обрабатывает пропуски в данных.
    - _check_duplicates — обрабатывает дубликаты: как по всей строке, так и по столбцу с id.
    - _check_min_words — подсчитывает число слов в тексте, создаёт столбец с этим числом и возвращает отфильтрованный DataFrame.

    публичные  методы :
    Публичные методы:
    - clean — последовательно применяет _check_missing, _check_duplicates, _check_min_words, возвращает очищенный DataFrame в атрибуте self.df_clean.
    - save_csv_or_pickle — сохраняет DataFrame  формате CSV или pickle.( указать имя без  расширения)
    - list_text - Создаёт в классе атрибут self.list_text со списком текстов из self.df_clean и возвращает его.
                   Если self.df_clean отсутствует, пишет предупреждение в лог.
    '''
    def __init__(self,  path:  Path, logger, min_words: int = 20 ):
        self.path = path
        self.logger = logger
        self.min_words = min_words
        self.df = pd.DataFrame(self._load_json(path))

    @staticmethod
    def _load_json(path: Path) -> dict:
         '''
        Загружает данные из JSON-файла.
        Args:
            path (Path): Путь к JSON-файлу.
        Returns:
            dict: Содержимое JSON-файла в виде словаря.
        '''    
         with open(path, 'r', encoding='utf-8') as file:
            return json.load(file)
         

    @staticmethod
    def _check_missing(df: pd.DataFrame,logger ) -> pd.DataFrame:
        '''    
        Returns:
            pd.DataFrame: DataFrame без строк с пропусками в 'uid' и 'text', с заполненными пропусками 'ru_wiki_pageid' значением 'unknown'.
        '''    
        col1, col2, col3 = df.columns

        if df.isna().any(axis=1).sum() == 0:
            logger.info('Пропуски не найдены')
            return df

        # Строки с пропусками в col1 или col3
        mask = df[[col1, col3]].isna().any(axis=1)
        if mask.sum() > 0:
            indices = df.index[mask].tolist()
            logger.warning(f'Строки с пропусками в столбцах {col1} или {col3} удалены: индексы {indices}')
            df = df.drop(df.index[mask])

        # Пропуски в col2
        mask = df[col2].isna()
        if mask.sum() > 0:
            indices = df.index[mask].tolist()
            logger.info(f'Пропуски в столбце {col2} заполнены значением "unknown", индексы: {indices}')
            df[col2] = df[col2].fillna('unknown')

        return df  
    
    @staticmethod
    def _check_duplicates(df: pd.DataFrame, logger) -> pd.DataFrame:
        '''
        Returns:
            pd.DataFrame: DataFrame без дубликатов по всем столбцам и по первому столбцу.'
        '''
        col1 = df.columns[0] 
        # Дубликаты во всём DataFrame (все столбцы)
        duplicates = df[df.duplicated()]
        if  len(duplicates)  > 0:
            indices = duplicates.index.tolist()
            logger.warning(f'Дубликаты удалены во всём DataFrame: индексы {indices}')
            df = df.drop_duplicates()
        else:
            logger.info('Дубликаты не найдены')

        # Дубликаты в столбце col1
        duplicates = df[df.duplicated(subset=[col1])]
        if  len(duplicates)  > 0:
            indices = duplicates.index.tolist()
            logger.warning(f'Дубликаты в столбце {col1} удалены: индексы {indices}')
            df = df.drop_duplicates(subset=[col1])
        else:
            logger.info(f'Дубликаты в столбце {col1} не найдены')

        return df    
        

    @staticmethod
    def _check_min_words(df: pd.DataFrame, logger, min_words) -> pd.DataFrame:
        '''
         Returns:
            pd.DataFrame: Отфильтрованный DataFrame по минимальному количеству слов  min_words  с добавленным столбцом 'count_words'
        '''
        def count_words(text: str) -> int:
            pattern = r'\b[а-яА-ЯёЁa-zA-Z]+\b'
            return len(re.findall(pattern, text))
        
        col_text = df.columns[-1]
        df[col_text] = df[col_text].astype(str)
        df['count_words'] = df[col_text].map(count_words)
     
        mask_short_text = df['count_words'] < min_words
        short_text_count = mask_short_text.sum()

        logger.warning(f'Удалено {short_text_count} строк с количеством слов меньше {min_words}')
        return df[~mask_short_text]

    def clean (self)-> pd.DataFrame:
        '''
        Returns:
            pd.DataFrame: Очищенный DataFrame, сохранённый в атрибуте self.df_clean.
        '''    
        df = self.df.copy()
        logger = self.logger
        df = self._check_missing(df, logger)
        df = self._check_duplicates(df, logger)
        self.df_clean = self._check_min_words(df, logger, self.min_words)
        return self.df_clean
    
    def list_texts(self) -> List:
        logger = self.logger
        if hasattr(self, 'df_clean'):
            self.list_text = [i for i in self.df_clean.text]
            return self.list_text
        else:
            self.logger.warning(f"Отсутствует атрибут 'df_clean' в объекте {self} — подготовленный DataFrame отсутствует.")
            self.logger.warning_console(f"Отсутствует атрибут 'df_clean' в объекте {self} — подготовленный DataFrame отсутствует.")    
    
    def save_csv_or_pickle(self, df: pd.DataFrame, name: str, format: Literal['csv', 'pickle'] = 'csv') -> None:
        '''
        Сохраняет DataFrame в указанный путь в формате CSV или pickle.

        Args:
            df (pd.DataFrame): DataFrame для сохранения.
            name  - имя  без расширения.
            format (Literal['csv', 'pickle'], optional): Формат файла. По умолчанию 'csv'.
        '''
        
        logger = self.logger
        if format not in ['csv', 'pickle']:
           format = 'csv'
        path = Path().cwd()/'data'/name
        path = path.with_suffix('.' + format)   
        if format == 'csv':
            
            df.to_csv(path, index=False)
        elif format == "pickle":
            df.to_pickle(path)
        logger.info(f'Файл сохранён. имя {path},  атрибут {df} (формат: {format})')


    def save_list_texts(self, name: str) -> None:
        logger = self.logger
        if hasattr(self, 'df_clean'):
            list_text = [i for i in self.df_clean.text]
            path = Path().cwd() / 'data' / name
            path = path.with_suffix('.pkl')
            with open(path, 'wb') as f:
                pickle.dump(list_text, f)
        else:
            self.logger.warning(f"Отсутствует атрибут 'df_clean' в объекте {self} — подготовленный DataFrame отсутствует.")
            self.logger.warning_console(f"Отсутствует атрибут 'df_clean' в объекте {self} — подготовленный DataFrame отсутствует.")