

import numpy as np

from openiva.commons.facial import get_transform_mat, warp_img, l2_norm, face_distance, sub_feature

MEAN_PTS_5 = np.array([[0.34191607, 0.46157411],
                       [0.65653392, 0.45983393],
                       [0.500225, 0.64050538],
                       [0.3709759, 0.82469198],
                       [0.63151697, 0.82325091]])

INDS_68_5 = [36, 45, 30, 48, 54]
