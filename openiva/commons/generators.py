import os
import traceback

from tqdm import tqdm

import cv2
import numpy as np

from .videocoding import decode_video_batch_local

from .io import imread


def read_images_local(pathes_imgs: list, batch_size=8, shuffle=False):
    q_dict_out = {'flag_start': True, 'flag_end': False, 'real_len': 0}

    samples_indecies = np.asarray(range(len(pathes_imgs)))
    if shuffle:
        np.random.shuffle(samples_indecies)

    last_i = samples_indecies[-1]

    pathes_imgs = {i: j for i, j in enumerate(pathes_imgs)}

    batch_images = []
    batch_indecies = []
    batch_pathes = []

    nb_samples = 0
    nb_batches = 0

    empty_frame = np.ones((100, 100, 3), dtype=np.uint8)

    for i_sample in tqdm(samples_indecies):
        nb_samples += 1

        path_img = pathes_imgs[i_sample]
        try:
            img = imread(path_img)
        except FileNotFoundError:
            traceback.print_exc()
            continue

        batch_images.append(img)
        batch_indecies.append(i_sample)
        batch_pathes.append(path_img)
        q_dict_out['real_len'] += 1

        if len(batch_images) == batch_size:
            # print('putting')

            q_dict_out['flag_end'] = (i_sample == last_i)
            q_dict_out['batch_src_size'] = [img.shape[:2][::-1]
                                            for img in batch_images]
            q_dict_out['batch_images'] = batch_images
            q_dict_out['batch_indecies'] = batch_indecies
            q_dict_out['batch_pathes'] = batch_pathes

            nb_batches += 1

            # q_compute.put(q_dict_out)
            yield q_dict_out

            # Flush

            batch_images = []
            batch_indecies = []
            batch_pathes = []
            q_dict_out = {'flag_start': False,
                          'flag_end': False, 'real_len': 0}

    q_dict_out['flag_end'] = True

    # Padding last batch to max batch size
    if q_dict_out['real_len']:
        print('The end batch real size is:{}'.format(q_dict_out['real_len']))
        if (not q_dict_out['real_len'] == batch_size):
            for _ in range(batch_size-q_dict_out['real_len']):
                batch_images.append(empty_frame)
                batch_indecies.append(i_sample)
                batch_pathes.append(path_img)

        q_dict_out['flag_start'] = (nb_samples == 1)
        q_dict_out['batch_src_size'] = [img.shape[:2][::-1]
                                        for img in batch_images]
        q_dict_out['batch_images'] = batch_images
        q_dict_out['batch_indecies'] = batch_indecies
        q_dict_out['batch_pathes'] = batch_pathes

        yield q_dict_out
