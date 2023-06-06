import atexit
import os
from datetime import datetime

import openai as openai
import pyminizip
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

if __name__ == "__main__":
    dev_env = os.environ.get('DEV_ENV', None)
    if not dev_env:
        zip_file = 'imitation-app.zip'
        ts = os.path.getmtime(zip_file)
        ts = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        print(f'decompressing zip file last modified at: {ts}')
        pyminizip.uncompress(zip_file, os.environ["ZIP_PWD"], None, 0)
    mod = __import__("gradio_app_builder")
    gradio_app = mod.build_app()
    atexit.register(mod.shutdown)
    gradio_app.launch(debug=not dev_env, share=dev_env is not None)
