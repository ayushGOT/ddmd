import inspect
import os 
import time
import yaml
import json
import logging
from typing import Union
from pathlib import Path
from typing import Type, TypeVar
from pydantic import BaseSettings as _BaseSettings

PathLike = Union[str, Path]
_T = TypeVar("_T")


def get_logger(debug=1):
    logger_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=logger_level, format='%(asctime)s %(message)s')
    logger = logging.getLogger(__name__)
    return logger


class BaseSettings(_BaseSettings):
    def dump_yaml(self, cfg_path: PathLike) -> None:
        with open(cfg_path, mode="w") as fp:
            yaml.dump(json.loads(self.json()), fp, indent=4, sort_keys=False)

    @classmethod
    def from_yaml(cls: Type[_T], filename: PathLike) -> _T:
        with open(filename) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)  # type: ignore[call-arg]


class yml_base(object): 
    def dump_yaml(self, cfg_path: PathLike) -> None: 
        with open(cfg_path, 'w') as yaml_file:
            yaml.dump(self.get_setup(), yaml_file, default_flow_style=False)


def create_path(sys_label=None, dir_type='md'): 
    """
    create MD simulation path based on its label (int), 
    and automatically update label if path exists. 
    """
    time_label = int(time.time())
    if sys_label: 
        md_path = f'{dir_type}_run_{sys_label}_{time_label}'
    else: 
         md_path = f'{dir_type}_run_{time_label}'

    try:
        os.mkdir(md_path)
        return md_path
    except: 
        return create_path(sys_label=sys_label, dir_type=dir_type)


def get_dir_base(file_path): 
    return os.path.basename(os.path.dirname(file_path))


def touch_file(file): 
    """
    create an empty file for bookkeeping sake
    """
    with open(file, 'w'): 
        pass


def get_numoflines(file): 
    return sum(1 for _ in open(file, 'r'))


def get_function_kwargs(func): 
    sig = inspect.signature(func)
    return [i for i in sig.parameters]


def dir_basename(path:str) -> str: 
    return os.path.basename(os.path.dirname(path))