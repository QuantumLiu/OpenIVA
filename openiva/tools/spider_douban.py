# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 10:19:27 2018

@author: quantumliu
"""

import os
import re
import traceback

import math

import shutil

from multiprocessing import Pool, cpu_count, freeze_support

import requests

HEADERS_TEMPLATE = {'accept': "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
                    "accept-encoding": "gzip, deflate, br",
                    "referer": "https://movie.douban.com/celebrity/{id_celeba}/photo/{id_image}/",
                    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36",
                    "upgrade-insecure-requests": "1"}

HEADERS_NORMAL = {'accept': "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
                  "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36"}

GENDER_DICT = {'男': 'm', '女': 'f'}

DIR_ROOT = os.path.dirname(os.path.dirname(__file__))

DEFAULT_IMGS_DIR = os.path.join(DIR_ROOT, 'datas/celebrity_images')


def _get_gender(url_celeba):

    res = requests.get(url_celeba, headers=HEADERS_NORMAL)
    text = res.text

    p_gender = r'<span>性别</span>: \n        (.*?)\n'

    gender_match = re.findall(p_gender, text)
    if len(gender_match) > 0:
        gender_zh = gender_match[0]
    else:
        gender_zh = ''
    gender_char = GENDER_DICT.get(gender_zh, 'm')

    return gender_char


def _parse_celeba_tuple_raw(celeba_tuple_raw):
    url, title = celeba_tuple_raw

    id_celeba = int(url.split('/')[-2])
    name = title.split(' ')[0]
    gender_char = _get_gender(url)

    return (name, id_celeba, url, gender_char)


def _get_celebrities_list(url_subject):
    url_celeba = url_subject+'/celebrities'
    res_celeba = requests.get(url_celeba, headers=HEADERS_NORMAL)
    src_celeba = res_celeba.text

    p_celeba = re.compile(
        r'<li class="celebrity">[\s\S]+?<div class="info">[\s\S]+?<a href="(.+?)" title="(.+?)"[\s\S]+?</div>')
    raw_info_celeba = re.findall(p_celeba, src_celeba)
    celebrities_list = [_parse_celeba_tuple_raw(ct) for ct in raw_info_celeba]
    return celebrities_list


def _get_all_turning_url(src_photos, max_img):
    p_turning = r'<span class="thispage" data-total-page="\d*?">1</span>([\s\S]*?)<span class="next">'
    match_result = re.findall(p_turning, src_photos)

    if len(match_result) > 0:
        src_turning = match_result[0]
    else:
        return []

    p_turning_url = r'<a href="(.*?)" >(\d*?)</a>'
    vis_turning_urls = re.findall(p_turning_url, src_turning)

    p_break = '<span class="break">...</span>'
    if not len(re.findall(p_break, src_turning)):
        return list(zip(*vis_turning_urls))[0]

    max_page = int(vis_turning_urls[-1][-1])

    p_step = ";start=(.*?)&"
    step = int(re.findall(p_step, vis_turning_urls[0][0])[0])

    max_page = min(math.ceil(max_img/step), max_page)

    temp = re.sub(r"start=\d*?&", 'start={n}&', vis_turning_urls[0][0])

    all_turning_url = [temp.format(n=n*step) for n in range(max_page)]
    return all_turning_url


def _get_id_image(src_photos):
    p_url = '<div class="cover">[\s\S]*?<a href="(.*?)" class=".*?">'
    urls = re.findall(p_url, src_photos)
    ids = [int(url.split('/')[-2]) for url in urls]

    return ids


def _format_raw_url(id_image):
    return 'https://img3.doubanio.com/view/photo/raw/public/p{}.jpg'.format(id_image)


def _format_headers(id_image, id_celeba):
    headers = HEADERS_TEMPLATE.copy()
    headers['referer'] = headers['referer'].format(
        id_image=id_image, id_celeba=id_celeba)
    return headers


def _display_search_result(celeba_tuple):
    name, id_celeba, url, gender_char = celeba_tuple
    print('Found {}, id: {}, gender: {}'.format(name, id_celeba, gender_char))


def _get_raw_photo_urls(id_celeba, max_img):
    url_root = 'https://movie.douban.com/celebrity/{}/photos/'.format(
        id_celeba)

    src_root = requests.get(url_root, headers=HEADERS_NORMAL).text
    turning_urls = _get_all_turning_url(src_root, max_img)

    src_all_photo = src_root + \
        ''.join(
            [requests.get(url, headers=HEADERS_NORMAL).text for url in turning_urls])

    ids_image = _get_id_image(src_all_photo)
    urls_and_headers = [(_format_raw_url(iid), _format_headers(
        iid, id_celeba)) for iid in ids_image]
    return urls_and_headers


def _download(root_dir, raw_photo_url, headers):
    fname = raw_photo_url.split('/')[-1]
    out_path = os.path.join(root_dir, fname)
    try:
        res = requests.get(raw_photo_url, headers=headers)
        data = res.content
        with open(out_path, 'wb') as fp:
            fp.write(data)
    except (requests.exceptions.BaseHTTPError, IOError, requests.exceptions.Timeout):
        traceback.print_exc()
        print('Got error during download img \n{img}\n to file \n{fname}\nPass'.format(
            img=raw_photo_url, fname=raw_photo_url))


def worker_celeba(root_dir, celeba_tuple, max_img=100, overwrite=False):
    _display_search_result(celeba_tuple)
    name, id_celeba, url, gender_char = celeba_tuple

    target_dir = os.path.join(
        root_dir, name+'_'+str(id_celeba)+'_'+gender_char)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    elif overwrite:
        shutil.rmtree(target_dir)
        os.mkdir(target_dir)
    else:
        return

    urls_and_headers = _get_raw_photo_urls(id_celeba, max_img)
    for raw_photo_url, headers in urls_and_headers[:max_img]:
        print('Downloading image of {name} from {url_img} ...'.format(
            name=name, url_img=raw_photo_url))
        _download(target_dir, raw_photo_url, headers)


class SearchNoResult(Exception):
    pass


def crawl_celeba(root_dir, name, max_img=200, overwrite=False):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    url_celeba_search = 'https://movie.douban.com/celebrities/search?search_text={}'.format(
        name)
    res = requests.get(url_celeba_search, headers=HEADERS_NORMAL)
    src_search = res.text

    p_id_celeba = r'<div class="result">[\s\S]+?<div class="content">[\s\S]+?<h3><a href="(.+?)" class="">(.*?)</a></h3>'

    results = re.findall(p_id_celeba, src_search)

    if len(results) < 1:
        raise SearchNoResult(
            'Got no result when searching actor {}.'.format(name))

    celeba_tuple = _parse_celeba_tuple_raw(results[0])

    worker_celeba(root_dir, celeba_tuple, max_img, overwrite)


def crawl_cast_s(root_dir, url_subject, max_img=100, overwrite=False):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    celebrities_list = _get_celebrities_list(url_subject)
    for celeba_tuple in celebrities_list:
        worker_celeba(root_dir, celeba_tuple, max_img, overwrite)


def crawl_cast_p(root_dir, url_subject, max_img=100, overwrite=False):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    celebrities_list = _get_celebrities_list(url_subject)

    mpool = Pool(min(4, cpu_count()))

    for celeba_tuple in celebrities_list:
        mpool.apply_async(
            worker_celeba, (root_dir, celeba_tuple, max_img, overwrite))
    mpool.close()
    mpool.join()


def crawl_cast(root_dir, url_subject, max_img=100, overwrite=False, use_par=False):
    if use_par:
        crawl_cast_p(root_dir, url_subject, max_img, overwrite)
    else:
        crawl_cast_s(root_dir, url_subject, max_img, overwrite)


def get_cast(url_subject):
    url_celeba = url_subject+'/celebrities'
    res_celeba = requests.get(url_celeba, headers=HEADERS_NORMAL)
    src_celeba = res_celeba.text

    p_celeba = re.compile(
        r'<li class="celebrity">[\s\S]+?<div class="info">[\s\S]+?<a href="(.+?)" title="(.+?)"[\s\S]+?</div>')
    raw_info_celeba = re.findall(p_celeba, src_celeba)
    cast = [_parse_celeba_tuple_raw(ct)[0] for ct in raw_info_celeba]
    return cast


if __name__ == '__main__':
    freeze_support()
    crawl_celeba('./datas/celebrity_images', '森高千里', overwrite=True)

    crawl_cast_p('./datas/celebrity_images',
                 'https://movie.douban.com/subject/3742360', overwrite=True)
