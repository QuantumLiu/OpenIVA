import os

import json

import numpy as np

import h5py


class FacialDB(object):
    '''
    人脸信息数据库
    Facial info database
    存储人脸对应的姓名、性别、特征向量等，可执行查询方法
    '''
    _DB_DICT = {}

    def __init__(self, path_json=None, db_dict=None) -> None:
        if not path_json is None:
            self.load_json(path_json)
        if not db_dict is None:
            self.update(db_dict)

    def query_N2N(self, features_tocheck, known_features=None, threshold=0.6):
        features_tocheck = np.ascontiguousarray(features_tocheck)
        known_features = (
            self.known_mean_features if known_features is None else known_features)

        dists_N = np.dot(features_tocheck, known_features)

        dists_max = dists_N.max(axis=-1)
        inds = dists_N.argmax(axis=-1)
        knowns = dists_max > threshold

        return inds, knowns, dists_max

    def query(self, feature_tocheck, known_features=None, threshold=0.6):
        inds, knowns, dists_max = self.query_N2N(
            [feature_tocheck], known_features, threshold)
        return inds[0], knowns[0], dists_max[0]

    @ property
    def db_dict(self):
        return self._DB_DICT.copy()

    @property
    def known_mean_features(self):
        list_features = [d["feature_vector"] for d in self._DB_DICT.values()]
        return np.ascontiguousarray(list_features).T

    @property
    def index2id(self):
        return {ind: id_person for ind, id_person in enumerate(self._DB_DICT.keys())}

    @property
    def id2name(self):
        return {k: v["name"] for k, v in self._DB_DICT.items()}

    @property
    def ind2name(self):
        return {ind: v["name"] for ind, v in enumerate(self._DB_DICT.values())}

    @property
    def all_names(self):
        return [v["name"] for v in self._DB_DICT.values()]

    @property
    def nb_people(self):
        return len(self._DB_DICT.keys())

    def append(self, id, info_dict):
        self._DB_DICT.update({id: info_dict})

    def update(self, db_dict):
        self._DB_DICT.update(db_dict)

    def load_json(self, path_json):
        with open(path_json, "r") as fp:
            db_dict = json.load(fp)
        self.update(db_dict)

    def save_to_json(self, path_json):
        with open(path_json, "w") as fp:
            json.dump(self.db_dict)
