import datetime
import logging
import random
import uuid
from threading import Lock
from typing import Optional, List, Tuple

from augmentation_service import AugmentationAgent
from index_service import IndexService
from model import SessionID
from openai_service import OpenAIClient
from parsing_service import RepliesParser
from prompts import MODIFIED_DAN_PROMPT, DEFAULT_CHAT_PROMPT, AGENTS, format_history, JOE_ROGAN_PROMPT
from util import log_execution

SYSTEM_MESSAGE_TEMPL = "Assistant role plays {agent}, a real live human."

SYSTEM_MESSAGES = {
    'enter_name': 'enter your name to begin conversation',
    'invalid_password': 'password must have at least 6 characters'
}


class Agent(object):
    def __init__(self, index_service: IndexService, augmentation_agent: AugmentationAgent):
        self.index_service = index_service
        self.augmentation_agent = augmentation_agent
        self.client = OpenAIClient()
        self.parser = RepliesParser()
        self.lock = Lock()

    # chat_inputs = [message_field, chat_state, username_field, agent_select]
    #         chat_outputs = [chat_area, chat_state, message_field]

    # def _select_chat_history(self, chat_history, index: GPTListIndex, max_tokens: int = 2200):
    #     messages = []
    #     tokens = 0
    #     if index:
    #         logging.debug('Index docs printout')
    #         for id, doc in index.docstore.docs.items():
    #             logging.debug(doc.text)
    #     for inp, reply in chat_history:
    #         user_msg_tokens = self.client.count_tokens(inp)
    #         agent_msg_tokens = self.client.count_tokens(reply)
    #         messages.append(())
    #         messages.append({"role": "user", "content": user_msg})
    #         if agent_msg and agent_msg_tokens + tokens < max_tokens:
    #             messages.append({"role": "assistant", "content": agent_msg})
    #             messages.append({"role": "user", "content": user_msg})
    #             tokens += (agent_msg_tokens + user_msg_tokens)
    #         else:
    #             logging.debug(f'Max history tokens limit reached: {tokens}')
    #             break
    #     for i, (role, msg) in enumerate(reversed(chat_history)):
    #
    #         user_msg_tokens = self.client.count_tokens(user_msg)
    #         agent_msg_tokens = self.client.count_tokens(agent_msg)
    #         if agent_msg and agent_msg_tokens + tokens < max_tokens:
    #             messages.append({"role": "assistant", "content": agent_msg})
    #             messages.append({"role": "user", "content": user_msg})
    #             tokens += (agent_msg_tokens + user_msg_tokens)
    #         else:
    #             logging.debug(f'Max history tokens limit reached: {tokens}')
    #             break
    #     return list(reversed(messages)), tokens

    # chat_inputs = [message_field, chat_state, username_field, agent_select, enter_password]
    # chat_outputs = [chat_area, chat_state, enter_password, message_field]

    @log_execution(cls_name='Agent')
    def __call__(
            self,
            inp: str,
            chat_history: List[Tuple[str, str]],
            user_name: Optional[str],
            agent_name: str,
            password: Optional[str]
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], str, str]:
        """Execute the chat functionality."""

        def filter_system_messages() -> List[Tuple[str, str]]:
            if not chat_history:
                return []

            def is_system_message(t):
                return t[1] in SYSTEM_MESSAGES.values()

            return list(filter(lambda t: not is_system_message(t), chat_history))

        if password and len(password) < 6:
            chat_history.append(('', SYSTEM_MESSAGES['invalid_password']))
            return chat_history, chat_history, password, inp,
        last_agent = None
        if password:
            last_agent = self.index_service.load_user(user_name, password)
            logging.info(f'loaded user. last agent {last_agent}, current agent: {agent_name}')
        else:
            password = str(uuid.uuid4())[:8]
        agent_name = agent_name or last_agent or random.choice(AGENTS.keys())
        chat_history = filter_system_messages()
        # if not chat_history or (len(chat_history) == 1 and chat_history[0][1] == 'enter a name to begin conversation'):
        #     chat_history = []
        if not user_name:
            chat_history.append(('', SYSTEM_MESSAGES['enter_name']))
            return chat_history, chat_history, password, inp,
        session_id = SessionID(username=user_name, agent=agent_name, password=password)

        # history = [(entry.user or '', entry.agent or '') for entry in chat_history]
        logging.debug('Waiting for a lock')
        self.lock.acquire()
        try:
            # def count_tokens():

            system_role = SYSTEM_MESSAGE_TEMPL.format(agent=agent_name)
            # message_history, history_tokens =  self._select_chat_history(chat_history,
            #                                                             self.index_service.get_chat_history(session_id))

            history_str = format_history(session_id, chat_history)
            history_tokens = self.client.count_tokens(history_str)
            hs = 0
            while history_tokens > 2200:
                hs -= 1
                history_str = format_history(session_id, chat_history[hs:])
                history_tokens = self.client.count_tokens(history_str)

            observations_str = self.augmentation_agent.get_observations(session_id, inp=inp)

            date_str = datetime.datetime.now().strftime('%a %b %-d %y')
            personality = AGENTS[agent_name]

            # logging.debug(history)
            # prompt_messages = message_history + [{"role": "user", "content": prompt, }]

            def get_chat_prompt(hint: str = '') -> Tuple[str, int]:
                prompt = None
                if agent_name == 'Dan':
                    prompt = MODIFIED_DAN_PROMPT.format(ai_prefix=agent_name, human_prefix=user_name,
                                                        dialogue_str=history_str,
                                                        observations_str=observations_str, date_str=date_str, input=inp,
                                                        hint=hint)
                elif agent_name == 'Joe Rogan':
                    prompt = JOE_ROGAN_PROMPT.format(human_prefix=user_name,
                                                     dialogue_str=history_str,
                                                     observations_str=observations_str, date_str=date_str, input=inp,
                                                     hint=hint)
                else:
                    prompt = DEFAULT_CHAT_PROMPT.format(ai_prefix=agent_name, personality=personality,
                                                        human_prefix=user_name,
                                                        observations_str=observations_str, date_str=date_str,
                                                        input=inp).strip()
                prompt_tokens = self.client.count_tokens(prompt)
                max_tokens = 4000 - history_tokens - prompt_tokens - 600
                logging.debug(f'Calculated max tokens: {max_tokens}')
                # logging.debug('-' * 40)
                logging.info('Prompt:')
                logging.info(prompt)
                return prompt, max_tokens

            # logging.debug('------ Selected Message history:-------')

            # logging.debug('-' * 40)

            def chat_with_correction(hint: str = None, temperature=0.7, use_backup: bool = False) -> Optional[str]:
                prompt, max_tokens = get_chat_prompt(hint)
                raw_reply = self.client.generate_chat_reply(
                    message_list=[{"role": "user", "content": prompt}],
                    system_role=system_role
                    , max_tokens=max_tokens, temperature=temperature
                )
                logging.info('--- Raw Reply ---')
                logging.info(raw_reply)
                ai_prefix = agent_name
                if agent_name == 'Joe Rogan':
                    ai_prefix = 'Joe'
                return self.parser.parse_chat_reply(raw_reply, ai_prefix, use_backup)

            reply = None
            hint = ''
            temp = 0.7
            for i in range(3):
                reply = chat_with_correction(hint, temp, i == 2)
                if reply:
                    break
                else:
                    temp -= 0.2
                    hint = f'This is your {i + 1} attempt of this prompt, follow the guidelines the best you can or let {agent_name} answer he is not comfortable answering and his reason for it'
            logging.info('--- Parsed Reply ---')
            logging.info(reply)
            chat_history.append((inp, reply))
            self.index_service.update(session_id, inp=inp, reply=reply)

            return chat_history, chat_history, password, ""
        except Exception:
            logging.exception('Agent exception')
            # traceback.print_exception(e)
            # todo error policy and debug
            error_reply = 'Oh no, I seem to be having connectivity issues...'
            chat_history.append((inp, error_reply))
            return chat_history, chat_history, password, ""
        finally:
            self.index_service.upload_index(session_id)
            logging.debug(f'{session_id} index uploaded, releasing lock')
            self.lock.release()
