import logging
import os
import pickle
import traceback
from logging import handlers
from os import listdir
from typing import Optional, Any

import gcsfs
import pyminizip

from util import log_execution


class GCPClient:

    def __init__(self):
        self._configure_logging()
        self.client = self.gcsfs_connect(cache_timeout=-1)
        self.bucket = os.environ['GCP_BUCKET']
        self.log_file_handler = None

    @staticmethod
    @log_execution(cls_name='GCPClient')
    def gcsfs_connect(cache_timeout=None) -> Optional[gcsfs.GCSFileSystem]:
        try:
            pyminizip.uncompress("gc.zip", os.environ["ZIP_PWD"], ".", 0)
            creds = None
            with open('creds.pkl', 'rb') as f:
                creds = pickle.load(file=f)
            os.remove('creds.pkl')
            return gcsfs.GCSFileSystem(project=os.environ['GCP_PROJECT'], token=creds, timeout=60,
                                       cache_timeout=cache_timeout)
            # print('GCSFS client initialized')
            # logging.info('GCSFS client initialized')
            # return system
        except Exception as e:
            logging.error(f'GCFS Exception {e}')
            traceback.print_exc()
            return None

    def _configure_logging(self):
        dev_env = os.environ.get('DEV_ENV', None)
        self.log_file_handler = handlers.TimedRotatingFileHandler(f"imitation-games{'-dev' if dev_env else ''}.log",
                                                                  when='d', interval=1,
                                                                  backupCount=3)
        log_level = os.environ.get('LOG_LEVEL', logging.INFO)
        dev_env = os.environ.get('DEV_ENV', None)

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(), self.log_file_handler]
        )
        logging.warning(f'Log level set to {logging.getLevelName(log_level)}. Dev env: {dev_env}')

    def upload_logs(self):
        # Resolve: ERROR:root:upload file imitationgames.log:  cannot schedule new futures after interpreter shutdown
        try:
            self.log_file_handler.close()
            logfiles = [f for f in listdir('.') if '.log' in f]
            for file in logfiles:
                # with open(file, 'r') as lf:
                #     data = lf.read()
                #     self.upload_str(file, data)
                # with self.client.open(f'{self.bucket}/{file}', 'w') as out:
                #     out.write(logfile.read())
                self.client.upload(file, f'{self.bucket}/logs/{file}')

            return True
        except Exception as e:
            logging.error(f'uploading log files failed:  {e}')
            return False
        pass

    def download_str(self, file) -> Optional[str]:
        try:
            with self.client.open(f'{self.bucket}/{file}', 'r') as f:
                return f.read()
        except Exception as e:
            logging.error(f'{file} not found: {e}')
            return None

    def upload_str(self, file: str, data: str) -> bool:
        try:
            with self.client.open(f'{self.bucket}/{file}', 'w') as f:
                f.write(data)
                return True
        except Exception as e:
            logging.error(f'upload file {file}:  {e}')
            return False

    def download_pkl(self, file) -> Optional[Any]:
        try:
            with self.client.open(f'{self.bucket}/{file}', 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logging.error(f'{file} not found: {e}')
            return None

    def upload_pkl(self, file, obj) -> bool:
        try:
            with self.client.open(f'{self.bucket}/{file}', 'wb') as f:
                pickle.dump(file=f, obj=obj)
                return True
        except Exception as e:
            logging.error(f'{file} upload failed: {e}')
            return False
