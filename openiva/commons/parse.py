# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 20:44:56 2021

@author: quantumliu
"""



import json

def parse_scene(result,thr_score=0.0,thr_split=200):
    '''
    解析物体识别
    thr_score:float,置信度阈值，大于它有效
    thr_split:int,判断连续片段阈值，出现目标帧数间隔大于该值，判断为新片段
    返回格式:
        {'人物名称':{'clips':[[idx1,idx2,...,idxn],...,[idx1,idx2,...,idxn]],
                    'scores':[[score1,score2,...,scoren],...,[score1,score2,...,scoren]]}
    '''
        
    result_scene=result['scene'].copy()
    
    info_scene={}
    for res_frame in result_scene:
        idx,names,scores=res_frame
        for name,score in zip(names[:3],scores[:3]):
            if score<thr_score:
                continue
            d=info_scene.setdefault(name,{'clips':[[]],'scores':[[]],'last':None})
            if d['last'] is None or (idx-d['last'])<=thr_split:
                d['clips'][-1].append(idx)
                d['scores'][-1].append(score)
            else:
                d['clips'].append([idx])
                d['scores'].append([score])
            d['last']=idx
    for v in info_scene.values():
        v.pop('last')
    
    return info_scene

def parse_facial(result,thr_score=0.6,thr_split=200):
    '''
    解析物体识别
    thr_score:float,置信度阈值，大于它有效
    thr_split:int,判断连续片段阈值，出现目标帧数间隔大于该值，判断为新片段
    返回格式:
        {'人物名称':{'uid':uid,
                    'clips':[[idx1,idx2,...,idxn],...,[idx1,idx2,...,idxn]],
                    'scores':[[score1,score2,...,scoren],...,[score1,score2,...,scoren]]
                    'rects':[[rect1,rect2,...,rectn],...,[rect1,rect2,...,rectn]]}
    '''
    result_facial=result['facial'].copy()
    
    
    
    
    info_facial={}
    for res_frame in result_facial:
        idx,flag_known,faces=res_frame
        for flag,(uid,name,_,score),rect in faces:
            
            if (not flag) or score<thr_score:
                continue
            d=info_facial.setdefault(name,{'clips':[[]],'scores':[[]],'rects':[[]],'last':None,'uid':uid})
            if idx in d['clips'][-1]:
                continue
            
            if d['last'] is None or (idx-d['last'])<=thr_split:
                d['clips'][-1].append(idx)
                d['scores'][-1].append(score)
                d['rects'][-1].append(rect)
            else:
                d['clips'].append([idx])
                d['scores'].append([score])
                d['rects'][-1].append([rect])
            d['last']=idx
    for v in info_facial.values():
        v.pop('last')
    
    return info_facial
    

def parse_obj(result,thr_score=0.4,thr_split=200):
    '''
    解析物体识别

    thr_score:float,置信度阈值，大于它有效
    thr_split:int,判断连续片段阈值，出现目标帧数间隔大于该值，判断为新片段
    返回格式:
        {'物体名称':{'clips':[[idx1,idx2,...,idxn],...,[idx1,idx2,...,idxn]],
                    'scores':[[[score1,score2,...,scoren],...,[score1,score2,...,scoren]],[[score1,score2,...,scoren],...,[score1,score2,...,scoren]]]
                    'rects':[[[rect1,rect2,...,rectn],...,[rect1,rect2,...,rectn]]，[[rect1,rect2,...,rectn],...,[rect1,rect2,...,rectn]]]}
    ###
    提醒:因为同一帧可以有多个同一种物体（比如同时好几个人），该返回值的rects和scores字段要多一层list
    ###
    '''
    result_obj=result['obj'].copy()
    
    
    
    
    info_obj={}
    for res_frame in result_obj:
        idx,flag_known,objs=res_frame
        if not flag_known:
            continue
        for name,score,rect in objs:
            
            if score<thr_score:
                continue
            d=info_obj.setdefault(name,{'clips':[[]],'scores':[[]],'rects':[[]],'last':None})
            
            if d['last'] is None or (idx-d['last'])<=thr_split:
                if idx in d['clips'][-1]:#此帧已经有同类物体
                    d['scores'][-1][-1].append(score)
                    d['rects'][-1][-1].append(rect)
                     
                else:
                    d['scores'][-1].append([score])
                    d['rects'][-1].append([rect])
                    d['clips'][-1].append(idx)

            else:
                d['clips'].append([idx])
                d['scores'].append([[score]])
                d['rects'].append([[rect]])
            d['last']=idx
    for v in info_obj.values():
        v.pop('last')
    
    return info_obj
    

if __name__=='__main__':
    with open('temp_multi.json','r') as fp:
        data_dict=json.load(fp)


    result=data_dict#['results']
    info_scene=parse_scene(result)
    info_facial=parse_facial(result)
    info_obj=parse_obj(result)
