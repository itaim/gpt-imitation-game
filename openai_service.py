from functools import lru_cache

import openai
import tiktoken
from retry import retry


class OpenAIClient(object):

    def __init__(self):
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

    @retry(tries=3, delay=1)
    def generate_chat_reply(self, message_list,
                            system_role: str = 'Role playing a human character in a dialogue with another person',
                            max_tokens=1024, temperature=0.6):
        system_message = [{
            'role': 'system',
            'content': system_role
        }
        ]
        response = openai.ChatCompletion.create(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            model="gpt-3.5-turbo",
            messages=system_message + message_list,
            # pl_tags=["imitation-game-1"]
        )
        return dict(list(dict(response.items())["choices"])[0])["message"]["content"]

    @retry(tries=3, delay=1)
    def get_embedding(self, text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]

    @lru_cache
    def count_tokens(self, message: str) -> int:
        if not message:
            return 0
        tokens = round(len(message.split()) * 1.28)
        try:
            return len(self.tokenizer.encode(message))
        except Exception as e:
            print(f'Tokenization failed {e}')
            return tokens
