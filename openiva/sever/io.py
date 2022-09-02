
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:44:09 2018

@author: quantumliu
"""

import os
import traceback
import hashlib
import json

import fcntl

import requests


from utils.pathes import RECEIVE_DIR,DB_PATH_RESOURCES
from utils.io import dump_result,get_video_info

def _read_chunks(file_handle, chunk_size=8192):
    while True:
        data = file_handle.read(chunk_size)
        if not data:
            break
        yield data

def checksum_md5(path):
    with open(path,'rb') as fp:
        hasher = hashlib.md5()
        for chunk in _read_chunks(fp):
            hasher.update(chunk)
    return hasher.hexdigest()

def download_video(url,out_path):
    try:
        res=requests.get(url)
        data=res.content
        print('Received data succesfully, saving to :{}'.format(out_path))
        with open(out_path,'wb') as fp:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
            fp.write(data)
        print('Save file completed')
    except Exception as e:
        traceback.print_exc()
        raise e

def get_resources_db():
    if not os.path.exists(DB_PATH_RESOURCES):
        db_dict= {}
    else:
        with open(DB_PATH_RESOURCES,'r') as fp:
            db_dict=json.load(fp)
    return db_dict

def update_resources_db(db_dict,uid,out_path):
    db_dict[uid]=out_path
    with open(DB_PATH_RESOURCES,'w') as fp:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        json.dump(db_dict,fp)
    #return db_dict
#def uid2path(db_path,ui):
    