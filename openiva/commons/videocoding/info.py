import cv2


def get_video_info(path) -> dict:
    cap = cv2.VideoCapture(path)
    nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    cap.release()

    duration = nb_frames/fps

    info_dict = {'duration': duration, 'nb_frames': nb_frames,
                 'fps': fps, 'width': width, 'height': height, 'fourcc': fourcc}
    return info_dict
