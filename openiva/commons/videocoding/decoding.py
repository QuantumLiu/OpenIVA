import numpy as np
import cv2

from tqdm import tqdm

from .info import get_video_info


def decode_video_batch_local(video_path: str, batch_size: int = 8, skip: int = 1):
    '''
    Data generator for decoding local videos and produce batch data.
    :returns
        yield batch data dictionary like:
        {'batch_images':[np.ndarray,...np.ndarray],
        'batch_indecies':[1,2,3,...],
        'batch_src_size':[(w,h),(w,h),...,(w,h)],
        'height':int
        'width': int
        'flag_start':bool,
        'flag_end':bool,'real_len':int}
    '''
    info_dict = get_video_info(video_path)

    cap = cv2.VideoCapture(video_path)

    nb_frames = info_dict['nb_frames']
    nb_samples = int(nb_frames/skip)
    samples_indecies = (np.asarray(range(nb_samples))*skip).tolist()
    samples_indecies = np.asarray(samples_indecies)
    total_samples = len(samples_indecies)

    i_frame = 0
    i_sample = 0

    last_i = samples_indecies[-1]
    # n_sample=0

    batch_images = []
    batch_indecies = []

    empty_frame = np.ones(
        (info_dict["height"], info_dict["width"], 3), dtype=np.uint8)

    q_dict_out = {'flag_start': True, 'flag_end': False, 'real_len': 0}

    for i_sample in tqdm(samples_indecies):
        if len(batch_images) == batch_size:
            # print('putting')

            q_dict_out['batch_src_size'] = [frame.shape[:2][::-1]
                                            for frame in batch_images]
            q_dict_out['height'], q_dict_out['width'] = frame.shape[:2]
            q_dict_out['batch_images'] = batch_images
            q_dict_out['batch_indecies'] = batch_indecies

            q_dict_out["info_dict"]=info_dict
            q_dict_out['flag_end'] = (i_sample == last_i)
            yield q_dict_out

            # Flush

            batch_images = []
            batch_indecies = []
            q_dict_out = {'flag_start': False,
                          'flag_end': False, 'real_len': 0}

        while True:
            flag, frame = cap.read()
            i_frame += 1
            if i_frame > i_sample:
                if not frame is None:
                    out = frame
                else:
                    Warning('Got empty frame {} in file {}'.format(
                        i_frame, video_path))
                    out = empty_frame
                batch_images.append(out)
                batch_indecies.append(i_frame-1)
                q_dict_out['real_len'] += 1
                break

    q_dict_out['flag_end'] = True

    # padding to max batch size
    if q_dict_out['real_len']:
        print('The end batch real size is:{}'.format(q_dict_out['real_len']))
        if (not q_dict_out['real_len'] == batch_size):
            for _ in range(batch_size-q_dict_out['real_len']):
                batch_images.append(empty_frame)
                batch_indecies.append(i_frame-1)
        q_dict_out['batch_src_size'] = frame.shape[:2][::-1]
        q_dict_out['height'], q_dict_out['width'] = out.shape[:2]
        q_dict_out['batch_images'] = batch_images
        q_dict_out['batch_indecies'] = batch_indecies
        q_dict_out["info_dict"]=info_dict
        yield q_dict_out
