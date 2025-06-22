
import numpy as np
from typing import Literal, List, Dict, Tuple
from llama_cpp import Llama

class LLMService:
    def __init__(self, model: Llama, config: Dict ) -> None:
        """
        Args:
            model (Llama): Экземпляр модели Llama для генерации текста.
            config (Dict): Конфигурация с параметрами модели и порогами.
        """
        self.model = model
        self.config = config

    def prompt_prepare(self, query: str, similarity_scores: np.ndarray, content: List[str]) -> str:
        """
        Формирует итоговый prompt для модели на основе запроса пользователя и оценок схожести.
        Args:
            query (str): Вопрос или запрос пользователя.
            similarity_scores (np.ndarray): Массив оценок схожести между запросом и контекстными документами.
            content (List[str]): Список текстов-контекстов для ответа.
        Returns:
            str: Сформированный prompt, который будет отправлен в LLM.
        """

        if (similarity_scores.size == 0 or np.min(similarity_scores) > self.config["threshold"]):
            prompt = f"""
            Вопрос: {query}
            Инструкции:
            1. Сообщи, что в базе данных нет информации по этому вопросу
            2. Будь максимально точным и кратким
            3. Сохраняй ясность и доброжелательность
            4. Допустимый вариант ответа:
              "Я не нашел информации по вашему запросу в доступных источниках."
              "Попробуйте уточнить вопрос или обратиться к другим материалам."
            5.Не предлагай собственных предположений или примерных ответов.
            6. В ответе не должно быть текста  из инструкции.
            7. Начни со слова Ответ:
            """
        else:

            context = []
            for i, score in enumerate(similarity_scores):
                # Проверим, что score для текущего вектора  меньше "threshold" и  рсхождение  с минимальным скором  меньше "distance_diff_vector"
                if score <= self.config["threshold"] and \
                   (score- similarity_scores.min()) <= self.config["distance_diff_vector"]:
                    context.append(content[i])
            context_text = "\n\n".join(context)   
            prompt = f"""
            Вопрос: {query}
            Контекст: {context_text}
            Инструкции:
            1. При ответе  бери информацию из контекста.
            2. Первый  абзац контекста  самый важный.
            3. Будь максимально точным и кратким
            3. Сохраняй ясность и доброжелательность
            4.Не предлагай собственных предположений или примерных ответов.
            """
        return prompt 

    def answer_question(self, prompt: str)-> str:
        """
        Args:
            prompt (str): Текст запроса для модели.
        Returns:
            str: Текст ответа, сгенерированный моделью.
        """
        output = self.model( prompt, ** self.config['config_LLM'])
        return output['choices'][0]['text']










