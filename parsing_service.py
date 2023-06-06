import logging
import re
from collections import defaultdict
from itertools import zip_longest
from typing import Optional

from model import AugmentationRequest

# , datefmt='%m/%d/%Y %I:%M:%S %p'
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')

logger = logging.getLogger('imitation')


class RepliesParser:

    def __init__(self):
        pass

    @staticmethod
    def parse_chat_reply(raw_reply: str, ai_prefix, use_backup: bool = False) -> Optional[str]:
        try:
            def pairs(t):
                it = iter(t)
                return zip_longest(*[it] * 2, fillvalue=None)

            regex = f'(AI:)|(Assistant:)|({ai_prefix}:)|(Reply:)'
            elements = iter(
                [el.strip() for el in re.split(regex, raw_reply) if
                 el])
            backup = ''
            for k, relevant in pairs(elements):
                if k.strip() == f'{ai_prefix}:':
                    logging.info(f'{ai_prefix} found. value: {relevant}')
                    match = re.match('.*agree.*with.*AI', relevant, flags=re.IGNORECASE)
                    if match and match.span()[0] < 20:
                        spos = relevant.index('.', match.span()[1])
                        if spos > -1:
                            return relevant[spos + 1:].strip()
                        else:
                            return relevant[match.span()[0] + 1:].strip()
                    return relevant
                match = re.match('(AI:)|(Assistant:)|(Reply:)', k, flags=re.IGNORECASE)
                if match:
                    logging.info(f' match on {k}:{relevant} adding as backup')
                    backup += relevant
            if use_backup:
                logging.info(f'Backup parsed reply: {backup}')
                return backup
            return None
        except Exception:
            logging.exception('parse chat reply')
            return None

    # ```
    # EMOTIONAL STATE: <{ai_prefix} emotional state in a word or two based on {ai_prefix} personality, chat history and {human_prefix} New message>
    # MEMORY: <optional, questions about {ai_prefix} and {human_prefix} as instructed above or None>
    # SEARCH: <optional, search queries if relevant to chat history, new {human_prefix} message or about events and developments {ai_prefix} or {human_prefix} are curious about or None>
    # ```
    @staticmethod
    def parse_augmentation_request(agent_reply: str) -> AugmentationRequest:
        try:
            def pairs(t):
                it = iter(t)
                return zip_longest(*[it] * 2, fillvalue=None)

            regex = f'(EMOTIONAL STATE:)|(MEMORY:)|(SEARCH:)|(Reply:)'
            elements = iter(
                [el.strip() for el in re.split(regex, agent_reply) if
                 el])
            data = defaultdict(list)
            for k, v in pairs(elements):
                if k.strip() == 'EMOTIONAL STATE:':
                    data['emotional state'] = v.strip()
                elif k.strip().lower() == 'reply:':
                    if len(v.split()) > 10:
                        data['reply'] = v
                else:
                    if 'None' not in v and 'N/A' not in v:
                        data[k.strip().strip(':').lower()] = [q.strip().strip('"') + '?' for q in v.split('?') if
                                                              q.strip().strip('"') and len(q.strip().split()) > 3]
            emotional = 'Neutral' if 'emotional state' not in data else data['emotional state']
            agent_reply = None if 'reply' not in data else data['reply']
            return AugmentationRequest(emotional_state=emotional, memory=data['memory'], search=data['search'],
                                       reply=agent_reply)
        except Exception:
            logging.exception(f'parse_augmentation_request: {agent_reply}')
            return AugmentationRequest(emotional_state='happy', memory=[], search=[], reply=None)


if __name__ == '__main__':
    raw1 = """AI: It's always sunny and pleasant in my digital world, but I'm not sure about the weather in the physical world. 
Dan: Hi Itai! The weather where I live is quite nice today. It's sunny with a light breeze, perfect for a walk outside."""
    parser = RepliesParser()
    print(parser.parse_chat_reply(raw1, 'Dan'))
