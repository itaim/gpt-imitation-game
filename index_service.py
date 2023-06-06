import asyncio
import logging
import random
import time
from collections import defaultdict
from typing import Optional

from gpt_index import GPTListIndex, Document
from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.langchain_helpers.chatgpt import ChatGPTLLMPredictor
from langchain.llms.openai import OpenAIChat

from model import SessionID
from openai_service import OpenAIClient
from prompts import AGENTS
from storage_service import GCPClient


# https://pypi.org/project/watchdog/
class IndexService(object):

    def __init__(self, storage: GCPClient):
        self.chat_history_dict = defaultdict(GPTListIndex)
        self.ai_client = OpenAIClient()
        self.storage_client = storage
        self.embed_model = OpenAIEmbedding()
        self.llm_predictor = ChatGPTLLMPredictor(llm=OpenAIChat())
        # asyncio.run(self.backup_indices())

    async def backup_indices(self):
        # await asyncio.sleep(5 * 60)
        # loop = asyncio.get_running_loop()
        # end_time = loop.time() + 5.0
        while True:
            self.upload_indices()
            logging.info(f'Indices backed up')
            # print(datetime.datetime.now())
            # if (loop.time() + 1.0) >= end_time:
            #     break
            await asyncio.sleep(5 * 60)

    def get_chat_history(self, state: SessionID) -> Optional[GPTListIndex]:
        key = self.history_key(state)
        return self.get_or_create_index(key)

    @staticmethod
    def history_key(session: SessionID) -> str:
        return f'{session.username}-{session.agent}-{session.password}'

    @staticmethod
    def history_filename(key: str, directory: str= 'history/') -> str:
        return f'{directory}history-{key}-li.json'

    def get_username_by_password(self, password: str) -> Optional[str]:
        for k in self.chat_history_dict.keys():
            parts = k.split('-')
            if parts[-1] == password:
                return parts[0]

    def load_user(self, username: str, pwd: str) -> str:
        agents_timestamps = []

        def load_index(agent):
            session = SessionID(username=username, agent=agent, password=pwd)
            key = self.history_key(session)
            index = self._load_index(key)
            if index:
                now = time.time()
                agents_timestamps.append(
                    max([(doc.extra_info.get('timestamp', now), agent) for doc in index.docstore.docs.values()],
                        key=lambda x: x[0]))
                self.chat_history_dict[key] = index

        for agent in AGENTS.keys():
            load_index(agent)

        if agents_timestamps:
            return max(agents_timestamps, key=lambda x: x[0])[1]
        else:
            return random.sample(AGENTS.keys(), k=1)[0]

    def _load_index(self, key, directory: str = 'history/') -> Optional[GPTListIndex]:
        filename = self.history_filename(key, directory)
        try:
            data = self.storage_client.download_str(file=filename)
            print(f'{filename} downloaded...')
            return GPTListIndex.load_from_string(data, embed_model=self.embed_model,
                                                 llm_predictor=self.llm_predictor)
        except:
            if directory == 'history/':
                return self._load_index(key, '')
            else:
                return None

    def upload_index(self, session_id):
        try:
            logging.debug(f'Uploading index to GCP')
            key = self.history_key(session_id)
            filename = self.history_filename(key)
            if self.storage_client.upload_str(filename, self.chat_history_dict[key].save_to_string()):
                logging.debug(f'{filename} uploaded')
        except Exception as e:
            logging.error(e)

    def upload_indices(self):
        try:
            logging.debug(f'Uploading indices to GCP')

            for key, index in self.chat_history_dict.items():
                filename = self.history_filename(key)
                if self.storage_client.upload_str(filename, index.save_to_string()):
                    logging.debug(f'{filename} uploaded')
        except Exception as e:
            logging.error(e)

    def update(self, session: SessionID, inp: str, reply: str):
        key = self.history_key(session)
        history = self.get_or_create_index(key)
        self.add_to_index(history, f'{session.first_name()}: {inp}')
        self.add_to_index(history, f'{session.agent}: {reply}')

    def get_or_create_index(self, key) -> GPTListIndex:
        index = self.chat_history_dict.get(key, None)
        if not index:
            index = GPTListIndex(documents=[], llm_predictor=self.llm_predictor, embed_model=self.embed_model)
            # index.set_text()
            # index.set_extra_info()
            # index.set_doc_id()
            self.chat_history_dict[key] = index
        return index

    def add_to_index(self, index: GPTListIndex, message: str):
        index.set_extra_info(extra_info={'last_message_timestamp': time.time()})
        index.insert(Document(text=message, embedding=self.ai_client.get_embedding(message),
                              extra_info={'timestamp': time.time()}))

    # def get_agent_index(self, state):
    #     return self.get_or_create_index(self.history_key(state.username,state.agent))
