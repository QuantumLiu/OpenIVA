import os
import traceback

import json

import numpy as np

GENDR_ID2NMAE = {0: "male", 1: "female", 2: "unknown"}


def _check_keys(func):
    def inner(*args):
        self = args[0]
        key = args[1]
        if not key in self._INFO_DICT.keys():
            raise KeyError(
                "Info key : {} is not aviliable in FacialInfo".format(key))
        return func(*args)
    return inner


class FacialInfo(object):
    _INFO_DICT = {"name": None, "id": None,
                  "gender_id": None, "feature_vector": None}

    def __init__(self, path_json=None, info_dict=None, ) -> None:
        if not path_json is None:
            self.load_info(path_json)
        elif not info_dict is None:
            self.update(info_dict)

    @property
    def info_dict(self):
        return self._INFO_DICT.copy()

    @_check_keys
    def set(self, key, value):

        self._INFO_DICT.update({key: value})

    @_check_keys
    def get(self, key):
        return self._INFO_DICT.get(key, None)

    def update(self, info_dict):
        self._INFO_DICT.update(info_dict)

    def load_info(self, path_json):
        with open(path_json, 'r') as fp:
            info_dict = json.load(fp)

        for key in info_dict.keys():
            if not key in self._INFO_DICT.keys():
                raise KeyError(
                    "Info key : {} is not aviliable in FacialInfo".format(key))

        self._INFO_DICT.update(info_dict)

        return self._INFO_DICT

    def save_info(self, path_json):
        try:
            with open(path_json, 'w') as fp:
                json.dump(self._INFO_DICT, fp)
        except:
            traceback.print_exc()
            os.remove(path_json)


def parse_filename(path_dir):
    name_dir = os.path.split(path_dir)[-1]

    name_celeba, id_douban, gender_char = name_dir.split('_')
    gender_id = {'m': 0, 'f': 1, 'u': 2}[gender_char]
    if id_douban.lower() == "n":
        id_douban
    return name_celeba, id_douban, gender_id


def remove_old(root_dir):
    pathes_dir = [path_dir for path_dir in [os.path.join(
        root_dir, name_dir) for name_dir in os.listdir(root_dir)] if os.path.isdir(path_dir)]
    for path_dir in pathes_dir:
        info_path = os.path.join(path_dir, 'info.json')
        if os.path.exists(info_path):
            os.remove(info_path)
