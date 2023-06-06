import pickle
from typing import Dict

import yaml
from yaml import load, dump, CDumper as dumper, CLoader as loader
from gpt_index import QueryMode


class AppConfig:
    def __init__(
            self,
            profile: str
    ):
        self.profile = profile

    def __repr__(self):
        return f"{self.__class__.__name__}(profile={self.profile}"


if __name__ == "__main__":
    # output_dir = "../config"
    #
    # configs = [
    #     AppConfig(
    #         "default"
    #     )
    # ]
    #
    # for conf in configs:
    #     output = dump(conf, Dumper=dumper)
    #     with open(f"{output_dir}/{conf.profile.lower()}.yaml", "w") as f:
    #         f.write(output)
    #
    #     with open(f"{output_dir}/{conf.profile.lower()}.yaml", "r") as f:
    #         obj = yaml.load(f.read(), loader)
    #     print(obj)
    with open('auth.pkl','wb') as f:
        pickle.dump({},f)
