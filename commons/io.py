import os

import traceback
import hashlib

import cv2
import numpy

import requests

def download_file(url,out_path,md5=None,overwite=False):
    try:
        if not os.path.exists(out_path) or overwite or (not md5 is None and not (checksum_md5(out_path)==md5)):
            print("Downloading file to {}".format(out_path))
            res=requests.get(url)
            data=res.content
            print('Received data succesfully, saving to :{}'.format(out_path))
            with open(out_path,'wb') as fp:
                fp.write(data)

            if not md5 is None:
                md5_local=checksum_md5(out_path)
                
                checked=(md5_local==md5)
                assert checked, "File {} MD5 check failed!".format(out_path)

            print('Save file completed')
        else:
            print("File {} already exists".format(out_path))


    except FileNotFoundError:
        traceback.print_exc()
        print("File {} path not exists".format(out_path))

    except AssertionError:
        traceback.print_exc()
        os.remove(out_path)
    

def checksum_md5(path):
    with open(path,'rb') as fp:
        hasher = hashlib.md5()
        for chunk in _read_chunks(fp):
            hasher.update(chunk)
    return hasher.hexdigest()
    
def _read_chunks(file_handle, chunk_size=8192):
    while True:
        data = file_handle.read(chunk_size)
        if not data:
            break
        yield data
