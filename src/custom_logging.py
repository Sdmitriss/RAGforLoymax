import logging
from pathlib import Path

class Customlogger:
    '''
    Кастомный логгер с разделением сообщений по уровню:

    - INFO и WARNING — записываются в один лог-файл.
    - ERROR — записывается в отдельный лог-файл.
    - Отдельный метод для вывода WARNING и ERROR в консоль.

    Аргументы конструктора:
    - log_dir (str): имя директории для логов (по умолчанию 'logs').
    - info_file (str): имя файла для INFO и WARNING (по умолчанию 'info.log').
    - error_file (str): имя файла для ERROR (по умолчанию 'error.log').
    - logger_name (str): имя логгера (по умолчанию 'logger_1').

    Основные методы:
    - info(msg): записывает информационное сообщение.
    - warning(msg): записывает предупреждение.
    - error(msg): записывает ошибку.
    - warning_console(msg): выводит предупреждение в консоль.
    - error_console(msg): выводит ошибку в консоль.
    '''
    def __init__(self, log_dir='logs', info_file='info.log', error_file='error.log', logger_name='logger_1'):
        base_dir = Path.cwd()
        log_dir_path = base_dir / log_dir
        log_dir_path.mkdir(exist_ok=True)

        info_path = log_dir_path / info_file
        error_path = log_dir_path / error_file

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)

        info_handler = logging.FileHandler(info_path, encoding='utf-8')
        info_handler.setLevel(logging.INFO)
        info_handler.addFilter(lambda record: record.levelno in (logging.INFO, logging.WARNING))
        info_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        info_handler.setFormatter(info_formatter)

        error_handler = logging.FileHandler(error_path,  encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        error_handler.setFormatter(error_formatter)

        # Консольный обработчик с уровнем INFO
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.console_handler.setFormatter(console_formatter)
        

        self.logger.addHandler(info_handler)
        self.logger.addHandler(error_handler)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def warning_console(self, msg):
        self.console_handler.setLevel(logging.WARNING)
        self.logger.addHandler(self.console_handler)
        self.logger.warning(msg)
        self.logger.removeHandler(self.console_handler)
        
    def error_console(self, msg):
        self.console_handler.setLevel(logging.ERROR)
        self.logger.addHandler(self.console_handler)
        self.logger.error(msg)
        self.logger.removeHandler(self.console_handler)    