import os
import fcntl

import time
import json

import traceback

import requests
from flask import Flask, request, make_response,Response

from threading import Thread
from queue import Queue
import uuid

from multiprocessing import Process
from multiprocessing import Queue as Queue_proc

from utils.pathes import DEFAULT_UNI_DB_PATH

Q_UPDATE=Queue_proc(100)

PATH_FLAG_UPDATE="/home/re02/pyprojects/VideoRecogInfer/datas/database/FLAG_UPDATE"

class InvalidUsage(Exception):
    status_code = 400
 
    def __init__(self, message, status_code=400):
        Exception.__init__(self)
        self.message = message
        self.status_code = status_code




SERVER_APP = Flask(__name__)
@SERVER_APP.errorhandler(InvalidUsage)
def _invalid_usage(error):
    response = Response(response=error.message,status=error.status_code)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return response

@SERVER_APP.route('/query_facial',methods=['GET'])
def query_facial():
    name=request.form.get('name',None)
    if name is None:
        raise InvalidUsage('Arguments "name" is empty')
    else:
        with open(DEFAULT_UNI_DB_PATH,'r') as fp:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
            info_dict=json.load(fp)
            fcntl.flock(fp.fileno(),fcntl.LOCK_UN)
        names=[d['name'] for d in info_dict.values()]

        response=Response(response=('true' if name in names else 'false'),status=200)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST'
        response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
        return response


@SERVER_APP.route('/start_update',methods=['GET'])
def start_update():
    callback_url=request.form.get('callback_url',None)
    if callback_url is None:
        raise InvalidUsage('Arguments "callback_url" is empty')
    else:
        Q_UPDATE.put({'callback_url':callback_url})

        response=Response(response='true',status=200)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST'
        response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
        return response

@SERVER_APP.route('/query_status',methods=['GET'])
def query_status():
    flag_update=os.path.exists(PATH_FLAG_UPDATE)

    response=Response(response=('true' if flag_update else 'false'),status=200)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return response

def _post_callback_update(callback_url,res):
    data_dict={'results':res}

    try:
        callback_res=requests.post(callback_url,data=data_dict)
        print('Callback request posted to {}'.format(callback_url))
    except (requests.ConnectionError,requests.Timeout):
        traceback.print_exc()
        
    return callback_res

class ThreadUpdate(Thread):
    def __init__(self,q_update):
        super(ThreadUpdate,self).__init__()
        self.q_update=q_update
    
    def run(self):
        from functions.register import register_all
        from utils.facial import save_facial_infos
        from utils.pathes import DEFAULT_UNI_DB_PATH,DEFAULT_IMGS_DIR

        print("Initing update process")
        while True:
            task=self.q_update.get()
            callback_url=task['callback_url']
            print('Starting update.')

            try:
                with open(PATH_FLAG_UPDATE,'w') as fp:
                    fp.write('start')

                results=register_all(DEFAULT_IMGS_DIR,DEFAULT_UNI_DB_PATH)
                res=[(name_celeba,'false') for (name_celeba,*_,flag_empty) in results if flag_empty]
                _post_callback_update(callback_url,res)
                print('Finished update.')
            finally:
                os.remove(PATH_FLAG_UPDATE)

def init_procs():
    q_update=Q_UPDATE
    proc_update=ThreadUpdate(q_update)
    proc_update.start()
