import numpy as np
import cv2


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


def get_transform_mat(image_landmarks, mean_pts, output_size=112, scale=1.0):
    if not isinstance(image_landmarks, np.ndarray):
        image_landmarks = np.array(image_landmarks)
    padding = 1  # (output_size / 64) * 1

    mat = umeyama(image_landmarks, mean_pts, True)[0:2]
    mat = mat * (output_size - 2 * padding)
    mat[:, 2] += padding
    mat *= (1 / scale)
    mat[:, 2] += -output_size*(((1 / scale) - 1.0) / 2)

    return mat


def transform_points(points, mat, invert=False):
    if invert:
        mat = cv2.invertAffineTransform(mat)
    points = np.expand_dims(points, axis=1)
    points = cv2.transform(points, mat, points.shape)
    points = np.squeeze(points)
    return points


def warp_img(mat, img, dshape=(112, 112), invert=False):
    if invert:
        M = cv2.invertAffineTransform(mat)
    else:
        M = mat
    warped = cv2.warpAffine(img, M, dshape, cv2.INTER_LANCZOS4)
    return warped


def l2_norm(x, axis=-1):
    """l2 norm"""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    output = x / norm

    return output


def face_distance(known_face_encoding, face_encoding_to_check):
    fl = np.asarray(known_face_encoding)
    return np.dot(fl, face_encoding_to_check)


def face_identify(known_face_encoding, face_encoding_to_check, tolerance=0.6):
    distance = face_distance(known_face_encoding, face_encoding_to_check)

    argmax = np.argmax(distance)
    d_min = distance[argmax]

    if distance[argmax] < tolerance:
        index = -1
        is_known = False
    else:
        index = argmax
        is_known = True
    return is_known, index, d_min


def sub_feature(feature_list, rate=0.9):
    feature_list = np.asarray(feature_list)
    mean_feature = np.mean(feature_list, axis=0)

    nb_feature = int(rate*len(feature_list))
    if nb_feature:
        dists = face_distance(feature_list, mean_feature)

        sub_feature_list = feature_list[np.argsort(dists)[::-1][:nb_feature]]
        mean_feature = l2_norm(np.mean(sub_feature_list, axis=0))
        return sub_feature_list, mean_feature
    else:
        return feature_list.copy(), feature_list[0].copy()
