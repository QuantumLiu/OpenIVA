import numpy as np
import cv2

from tqdm import tqdm

from .info import get_video_info

def decode_video_batch_local(video_path:str,batch_size:int = 8,skip:int = 1):
        info_dict=get_video_info(video_path)
        
        cap=cv2.VideoCapture(video_path)
        
        
        nb_frames=info_dict['nb_frames']
        nb_samples=int(nb_frames/skip)
        samples_indecies=(np.asarray(range(nb_samples))*skip).tolist()
        samples_indecies=np.asarray(samples_indecies)
        total_samples=len(samples_indecies)
        
        i_frame=0
        i_sample=0
        # n_sample=0
        batch_frames=[]
        batch_indecies=[]

        empty_frame=np.ones((info_dict["height"],info_dict["width"],3),dtype=np.uint8)

        q_dict_out={'flag_start':True,'flag_end':False,'real_len':0}

        for i_sample in tqdm(samples_indecies):
            if len(batch_frames)==batch_size:
                #print('putting')

                q_dict_out['src_size']=frame.shape[:2]
                q_dict_out['batch_frames']=batch_frames
                q_dict_out['batch_indecies']=batch_indecies

                # q_compute.put(q_dict_out)
                yield q_dict_out

                #Flush

                batch_frames=[]
                batch_indecies=[]
                q_dict_out={'flag_start':False,'flag_end':False,'real_len':0}
                
            while True:
                flag,frame=cap.read()
                i_frame+=1
                if i_frame>i_sample:
                    #print(i_frame)
                    if not frame is None:
                        out=frame#cv2.resize(frame,(224,224))
                    else:
                        Warning('Got empty frame {} in file {}'.format(i_frame,video_path))
                        out=empty_frame
                    batch_frames.append(out)
                    batch_indecies.append(i_frame-1)
                    q_dict_out['real_len']+=1
                    #print('append',i_frame-1)
                    break
         
        q_dict_out['flag_end']=True
        # q_dict_out['info_form']=q_dict_task['info_form']

        #padding to max batch size
        if q_dict_out['real_len'] :
            print('The end batch real size is:{}'.format(q_dict_out['real_len']))
            if (not q_dict_out['real_len']==batch_size):
                for _ in range(batch_size-q_dict_out['real_len']):
                    batch_frames.append(empty_frame)
                    batch_indecies.append(i_frame-1)
            #print('putting')
            q_dict_out['src_size']=out.shape[:2]
            q_dict_out['batch_frames']=batch_frames
            q_dict_out['batch_indecies']=batch_indecies
            yield q_dict_out