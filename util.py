import functools
import logging
import os
import time
from typing import List

import pyminizip


def get_app_files(dir: str = '.') -> List[str]:
    skip = {'app.py', 'util.py'}
    return [f for f in os.listdir(dir) if f.endswith('.py') and f not in skip]


def log_execution(cls_name=None):
    def log_decorator(func):
        @functools.wraps(func)
        def func_wrapper(*args, **kwargs):
            fname = f'{cls_name}.{func.__name__}' if cls_name else func.__name__
            logging.info(f"Executing {fname}")
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logging.info(f"{fname} took {end_time - start_time} seconds to run.")
            return result

        return func_wrapper

    return log_decorator


if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    print(f'compressing app files')
    pyminizip.compress_multiple(get_app_files(), [], 'imitation-app.zip', os.environ["ZIP_PWD"], 2)
