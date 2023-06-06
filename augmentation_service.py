import datetime
import logging
import pprint
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Optional

from gpt_index import GPTListIndex, OpenAIEmbedding
from gpt_index.indices.prompt_helper import PromptHelper
from gpt_index.indices.query.list import GPTListIndexEmbeddingQuery
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.response.builder import ResponseMode
from gpt_index.langchain_helpers.chatgpt import ChatGPTLLMPredictor
from langchain.agents import load_tools
from langchain.llms.openai import OpenAIChat
from tqdm import tqdm

from index_service import IndexService
from model import SessionID, AugmentationRequest, AugmentationResponse
from openai_service import OpenAIClient
from parsing_service import RepliesParser
from prompts import AGENTS, OBSERVATIONS_PROMPT, OBSERVATIONS_TEMPLATE, DEFAULT_TEXT_QA_PROMPT_TMPL


class AugmentationAgent:

    def __init__(self, index_service: IndexService, executor: ThreadPoolExecutor, verbose: bool = False):
        self.index_service = index_service
        self.gpt_client = OpenAIClient()
        self.serp = load_tools(['serpapi'])[0]
        self.executor = executor
        self.embed_model = OpenAIEmbedding()
        self.llm_predictor = ChatGPTLLMPredictor(llm=OpenAIChat())
        self.prompt_helper = PromptHelper.from_llm_predictor(
            self.llm_predictor, chunk_size_limit=3500
        )
        self.verbose = verbose
        self.parser = RepliesParser()

    def process_request(self, state: SessionID, observation: AugmentationRequest) -> AugmentationResponse:
        n_tasks = len(observation.search) + len(observation.memory)
        if not n_tasks:
            return AugmentationResponse(memory=[], search=[])
        with tqdm(total=n_tasks) as pbar:
            with ThreadPoolExecutor(max_workers=min(n_tasks, 8)) as executor:
                search = []
                memory = []
                for q in observation.search:
                    search.append(executor.submit(self._search, q))
                for q in observation.memory:
                    memory.append(executor.submit(self._query_history, q))

                # wait for all tasks to complete
                print('Waiting for tasks to complete...')
                wait(search + memory)
                print('All tasks are done!')
                search = list(filter(lambda r: r, map(lambda f: f.result(), search)))
                memory = list(filter(lambda r: r, map(lambda f: f.result(), memory)))

        return AugmentationResponse(memory=memory, search=search)

    def _query_index(self, index: GPTListIndex, query: str) -> Optional[str]:
        engine = GPTListIndexEmbeddingQuery(index_struct=index.index_struct, similarity_top_k=10,
                                            embed_model=self.embed_model, llm_predictor=self.llm_predictor,
                                            prompt_helper=self.prompt_helper, response_mode=ResponseMode.DEFAULT)
        nodes = engine.get_nodes_and_similarities_for_response(query_bundle=QueryBundle(query_str=query))
        if nodes:
            logging.info(f'\n{len(nodes)} nodes returned for query: {query}')
            contex_str = '\n'.join([node.text for node, sim in nodes])

            prompt = DEFAULT_TEXT_QA_PROMPT_TMPL.format(context_str=contex_str,
                                                        query_str=query).strip()
            message_list = [{"role": "system",
                             "content": f"Answer query from provided context"},
                            {"role": "user",
                             "content": prompt}]
            logging.info('--- Index Query ----')
            logging.info(prompt)
            reply = self.gpt_client.generate_chat_reply(message_list=message_list, temperature=0.0)
            if reply.startswith('Unknown'):
                return None
            else:
                return reply
        else:
            logging.info(f'\nNo nodes returned for query: {query}')
            return None

    def _query_history(self, state: SessionID, query: str) -> Optional[str]:
        index = self.index_service.get_chat_history(state)
        if index:
            return self._query_index(index, query)
        else:
            return None

    def _search(self, query: str) -> Optional[str]:
        try:
            return self.serp.run(query)
        except Exception:
            logging.exception(f'SERP exception')
            return None

    def get_observations(self, state: SessionID, inp: str) -> Optional[str]:
        # system_message = {
        #     "role": "system",
        #     "content": f"Assistant is role playing {state.agent}, a real live human.",
        # }
        # messages = [system_message] + history
        date_str = datetime.datetime.now().strftime('%a %b %-d %y')

        personality = AGENTS[state.agent]
        prompt = OBSERVATIONS_PROMPT.format(ai_prefix=state.agent, human_prefix=state.username,
                                            personality=personality, date_str=date_str,
                                            input=inp).strip()
        observations_reply = self.gpt_client.generate_chat_reply([{
            "role": "user",
            "content": prompt,
        }])
        logging.debug(f'observations_reply:\n{observations_reply}')
        observation_request = self.parser.parse_augmentation_request(observations_reply)
        print('Observation Request:')
        pprint.pp(observation_request)
        context = self.process_request(state, observation_request)
        augmentation_str = ''
        if context.memory:
            augmentation_str += '\n'.join(context.memory)
        if context.search:
            data = '\n'.join(context.search)
            augmentation_str += f'web search:\n{data}'
        if augmentation_str:
            augmentation_str = OBSERVATIONS_TEMPLATE.format(human_prefix=state.username,
                                                            observations_str=augmentation_str)
        # if state.agent != 'Dan':
        #     observations_str += f'emotional state: \n{observation_request.emotional_state}'

        logging.info(f'\n---- Augmentation Str -----\n{augmentation_str}')
        return augmentation_str or None
