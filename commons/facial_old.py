import os
import traceback

import json

import numpy as np
import cv2


GENDR_ID2NMAE={0:"male",1:"female",2:"unknown"}

def _check_keys(func):
    def inner(*args):
        self=args[0]
        key=args[1]
        if not key in self._INFO_DICT.keys():
            raise KeyError("Info key : {} is not aviliable in FacialInfo".format(key))
        return func(*args)
    return inner
        

class FacialInfo(object):
    _INFO_DICT={"name":None,"id":None,"gender_id":None,"feature_vector":None}
    def __init__(self,path_json=None, info_dict=None, ) -> None:
        if not path_json is None:
            self.load_info(path_json)
        elif not info_dict is None:
            self.update(info_dict)


    @property
    def info_dict(self):
        return self._INFO_DICT.copy()

    @_check_keys
    def set(self,key,value):
            
        self._INFO_DICT.update({key:value})

    @_check_keys
    def get(self,key):
        return self._INFO_DICT.get(key,None)

    def update(self,info_dict):
        self._INFO_DICT.update(info_dict)

    def load_info(self,path_json):
        with open(path_json,'r') as fp:
            info_dict=json.load(fp)

        for key in info_dict.keys():
            if not key in self._INFO_DICT.keys():
                raise KeyError("Info key : {} is not aviliable in FacialInfo".format(key))

        self._INFO_DICT.update(info_dict)

        return self._INFO_DICT
    
    def save_info(self,path_json):
        try:
            with open(path_json,'w') as fp:
                json.dump(self._INFO_DICT,fp)
        except:
            traceback.print_exc()
            os.remove(path_json)

def parse_filename(path_dir):
    name_dir=os.path.split(path_dir)[-1]
    
    name_celeba,id_douban,gender_char=name_dir.split('_')
    gender_id={'m':0,'f':1,'u':2}[gender_char]
    if id_douban.lower()=="n":
        id_douban
    return name_celeba,id_douban,gender_id

def remove_old(root_dir):
    pathes_dir=[path_dir for path_dir in [os.path.join(root_dir,name_dir) for name_dir in os.listdir(root_dir)] if os.path.isdir(path_dir)]
    for path_dir in pathes_dir:
        info_path=os.path.join(path_dir,'info.json')
        if os.path.exists(info_path):
            os.remove(info_path)


def umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = np.dot(dst_demean.T, src_demean) / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V.T))

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale

    return T


def get_transform_mat (image_landmarks,mean_pts, output_size=112, scale=1.0):
    if not isinstance(image_landmarks, np.ndarray):
        image_landmarks = np.array (image_landmarks) 
    padding = 1#(output_size / 64) * 1

    mat = umeyama(image_landmarks, mean_pts, True)[0:2]
    mat = mat * (output_size - 2 * padding)
    mat[:,2] += padding        
    mat *= (1 / scale)
    mat[:,2] += -output_size*( ( (1 / scale) - 1.0 ) / 2 )

    return mat

def transform_points(points, mat, invert=False):
    if invert:
        mat = cv2.invertAffineTransform (mat)
    points = np.expand_dims(points, axis=1)
    points = cv2.transform(points, mat, points.shape)
    points = np.squeeze(points)
    return points

def warp_img(mat,img,dshape=(112,112),invert=False):
    if invert:
        M=cv2.invertAffineTransform(mat)
    else:
        M=mat
    warped=cv2.warpAffine(img,M,dshape,cv2.INTER_LANCZOS4)
    return warped

def l2_norm(x, axis=-1):
    """l2 norm"""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    output = x / norm

    return output

def face_distance(known_face_encoding,face_encoding_to_check):
    fl=np.asarray(known_face_encoding)
    return np.dot(fl,face_encoding_to_check)

def face_identify(known_face_encoding, face_encoding_to_check, tolerance=0.6):
    distance=face_distance(known_face_encoding, face_encoding_to_check)
    
    argmax=np.argmax(distance)
    d_min=distance[argmax]

    if distance[argmax]<tolerance:
        index=-1
        is_known=False
    else:
        index=argmax
        is_known=True
    return is_known,index,d_min

def sub_feature(feature_list,rate=0.9):
    feature_list=np.asarray(feature_list)
    mean_feature=np.mean(feature_list,axis=0)
    
    nb_feature=int(rate*len(feature_list))
    if nb_feature:
        dists=face_distance(feature_list,mean_feature)
        
        sub_feature_list= feature_list[np.argsort(dists)[::-1][:nb_feature]]
        mean_feature=l2_norm(np.mean(sub_feature_list,axis=0))
        return sub_feature_list,mean_feature
    else:
        return feature_list.copy(),feature_list[0].copy()


