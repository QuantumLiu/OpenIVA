# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:43:04 2018

@author: quantumliu
"""


import time
import json

import traceback
import threading  
import queue  

import requests
from flask import Flask, request, make_response,Response

from multiprocessing import Queue,Process
import uuid

from worker.data import ProcessDataMulti2,ProcessFacialParse,ProcessCollect
from worker.multi import ProcessMulti2

from utils.pathes import RESNET50_365_MODEL_PATH_B1,YOLOV4_AD8_MODEL_PATH_B1,RESNET50_RETINAFACE_MODEL_PATH_B1
from utils.pathes import RESNET50_365_MODEL_PATH_B4,YOLOV4_AD8_MODEL_PATH_B4,RESNET50_RETINAFACE_MODEL_PATH_B4
from utils.pathes import RESNET50_365_MODEL_PATH_B8,YOLOV4_AD8_MODEL_PATH_B8,RESNET50_RETINAFACE_MODEL_PATH_B8
from utils.pathes import RESNET50_ARCFACE_MODEL_PATH_B1

from utils.io import dump_result

from models.yolo.utils import load_class_names

import time

# =============================================================================
# from multiprocessing import Pool,freeze_support,cpu_count
# import cv2
# import numpy as np
# =============================================================================

# =============================================================================
# from api.extern import cpp_detect
# from api.alignment import dlib_alignment
# from api.recognition import face_encoding,face_distance,face_identify
# 
# from api.worker import identify_worker
# =============================================================================
import os


from utils.io import get_video_info

from .io import checksum_md5,download_video,dump_result,get_resources_db,update_resources_db,RECEIVE_DIR

Q_TASK=Queue()
Q_COMPUTE_IN=Queue(300)
Q_COMPUTE_OUT=Queue()



class InvalidUsage(Exception):
    status_code = 400
 
    def __init__(self, message, status_code=400):
        Exception.__init__(self)
        self.message = message
        self.status_code = status_code




SERVER_APP = Flask(__name__)
@SERVER_APP.errorhandler(InvalidUsage)
def _invalid_usage(error):
    response = make_response(error.message)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return response

@SERVER_APP.route('/receive',methods=['POST'])
def receive_video_file():
    db=get_resources_db()

    file_name=request.form.get('file_name',None)
    print(file_name)
    file_url=request.form.get('file_url',None)
    print(file_url)
    md5_received=request.form.get('md5',None)
    
    overwrite=request.form.get('overwrite',False)

    uid=request.form.get('uid',None)
# =============================================================================
#     uid_sub=request.form.get('uid_sub',None)
#     url_sub=request.form.get('sub_url',None)
# =============================================================================
    
    if ( None in [file_url,md5_received]):
        raise InvalidUsage
    
    out_path=RECEIVE_DIR+file_name
    if (not os.path.exists(out_path)) or overwrite:
        download_video(file_url,out_path)
    update_resources_db(db,uid,out_path)

    md5_local=checksum_md5(out_path)
    
    checked=(md5_local==md5_received)
    
    result_dict={'received_flag':True,'checksum_flag':checked,'uid':uid}
    result_json=json.dumps(result_dict)
    
    response=Response(response=result_json,status=200)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return response

    
@SERVER_APP.route('/funcs_multi',methods=['POST'])
def run_multi_fuctions():
    db=get_resources_db()
    # str_func_types=request.form.get('func_types','[]')
    # print(str_func_types,type(str_func_types))
    # func_types=json.loads(str_func_types)
    # print(func_types,type(func_types))
    # if not len(func_types):
    #     raise InvalidUsage('No function type specified.')
        
    #md5_received=request.form.get('md5',None)
    

    uid=request.form.get('uid',None)
    uid_sub=request.form.get('uid_sub',None)
    url_sub=request.form.get('url_sub',None)
    callback_url=request.form.get('callback_url',None)
    out_fn=db.get(uid)
    
    if out_fn is None:
        raise InvalidUsage('Never recived resource uid: {}'.format(uid))
    out_path=os.path.join(RECEIVE_DIR,os.path.split(out_fn)[-1])
# =============================================================================
#     if md5_received is None:
#         raise InvalidUsage('Not recived md5.')
# =============================================================================

# =============================================================================
#     if (func_type is None) or func_type not in ['facial','obj','scene']:
#         raise InvalidUsage('No function type specified.')
# =============================================================================
        
# =============================================================================
#     md5_local=checksum_md5(out_path)
# =============================================================================
    
# =============================================================================
#     checked=(md5_local==md5_received)
# =============================================================================
    
    #result_dict={'received_flag':True,'checksum_flag':checked,'uid':uid}
    result_dict={'received_flag':True,'uid':uid}
    result_json=json.dumps(result_dict)
    
    #q_dict={'out_path':out_path,'func_types':func_types,'callback_url':callback_url,'uid':uid,'uid_sub':uid_sub,'url_sub':url_sub,'md5':md5_received}
    info_form={'out_path':out_path,'callback_url':callback_url,'uid':uid,'uid_sub':uid_sub,'url_sub':url_sub}
    q_dict={'video_path':out_path,'task_id':str(uuid.uuid1()),'info_form':info_form}
    print(q_dict)
    Q_TASK.put(q_dict)
        
    response=Response(response=result_json,status=200)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return response
    



def _post_callback_multi(video_info_json,result_json,callback_url,uid,uid_sub,url_sub):
    data_dict={'result_json':result_json,'info_json':video_info_json,'callback_url':callback_url,
               'received_flag':True,'checksum_flag':True,'uid':uid,
               'uid_sub':uid_sub,'url_sub':url_sub}
    
    try:
        callback_res=requests.post(callback_url,data=data_dict)
    except (requests.ConnectionError,requests.Timeout):
        traceback.print_exc()
        
    print('Callback request posted to {}'.format(callback_url))
    return callback_res

                
class ProcessCallback(Process):
    def __init__(self,q_post):
        super(ProcessCallback,self).__init__()
        self.q_post=q_post
        
    def start(self):
        super(ProcessCallback,self).start()
        print('Daemon process for multi ai functions started.')
        
    def run(self):
        while True:
            q_dict=self.q_post.get()
            task_id=q_dict['task_id']
            results=q_dict['results']
            info_form=q_dict['info_form']

            out_path=info_form['out_path']
            callback_url=info_form['callback_url']
            uid=info_form['uid']
            uid_sub=info_form['uid_sub']
            url_sub=info_form['url_sub']
            # func_types=info_form['func_types']
            
            video_info=get_video_info(out_path)
            video_info_json=dump_result(video_info)
            
            # print(func_types)
            result_json=dump_result(results)
            callback_res=_post_callback_multi(video_info_json,result_json,callback_url,uid,uid_sub,url_sub)
            with open('temp_multi.html','w') as fp:
                fp.write(callback_res.text)
            with open('temp_multi.json','w') as fp:
               fp.write(result_json)
            #print(callback_res.text)

def init_procs(batch_size=8,nb_proc=4,num_classes_yolo=8,path_names="/home/re02/pyprojects/VideoRecogInfer/datas/index/ad.names"):    
    if batch_size==1:
        yolo_path=YOLOV4_AD8_MODEL_PATH_B1
        places365_path=RESNET50_365_MODEL_PATH_B1
        retinaface_path=RESNET50_RETINAFACE_MODEL_PATH_B1
    elif batch_size==4:
        yolo_path=YOLOV4_AD8_MODEL_PATH_B4
        places365_path=RESNET50_365_MODEL_PATH_B4
        retinaface_path=RESNET50_RETINAFACE_MODEL_PATH_B4
    elif batch_size==8:
        yolo_path=YOLOV4_AD8_MODEL_PATH_B8
        places365_path=RESNET50_365_MODEL_PATH_B8
        retinaface_path=RESNET50_RETINAFACE_MODEL_PATH_B8
    
    arcface_path=RESNET50_ARCFACE_MODEL_PATH_B1

    model_configs={'yolov4':[yolo_path,(608,608),8,'yolov4','yolov4'],\
                    'places365':[places365_path,(224,224),365,'places365','places365'],\
                    'retinaface':[retinaface_path,(640,640),None,'retinaface','retinaface'],\
                    'arcface':[arcface_path,(112,112),None,'arcface','arcface']}

    class_names=load_class_names(path_names)

    q_task=Q_TASK
    q_compute_in=Q_COMPUTE_IN
    q_compute_out=Q_COMPUTE_OUT
    
    q_facial_in=Queue()
    q_facial_out=Queue()

    q_post=Queue()

    p_compute=ProcessMulti2(q_compute_in,q_compute_out,q_facial_in,q_facial_out,model_configs,batch_size)

    p_collect=ProcessCollect(q_compute_out,q_post,class_names)
    p_facial=ProcessFacialParse(q_facial_in,q_facial_out)

    p_datas=[ProcessDataMulti2(q_task,q_compute_in,model_configs,batch_size=batch_size,skip=10) for i in range(nb_proc)]
    
    p_post=ProcessCallback(q_post)


    p_compute.start()
    time.sleep(5)
    for p_data in p_datas:
        p_data.start()
    p_facial.start()
    p_collect.start()
    p_post.start()
