import numpy as np

from .loader import load_source_frames
from .source2info import Source2Info


def _mean_filter(arr, k):
    n = arr.shape[0]
    half_k = k // 2
    res = []
    for i in range(n):
        s = max(0, i - half_k)
        e = min(n, i + half_k + 1)
        res.append(arr[s:e].mean(0))
    res = np.stack(res, 0)
    return res


def smooth_x_s_info_lst(x_s_info_list, ignore_keys=(), smo_k=13):
    keys = x_s_info_list[0].keys()
    N = len(x_s_info_list)
    smo_dict = {}
    for k in keys:
        _lst = [x_s_info_list[i][k] for i in range(N)]
        if k not in ignore_keys:
            _lst = np.stack(_lst, 0)
            _smo_lst = _mean_filter(_lst, smo_k)
        else:
            _smo_lst = _lst
        smo_dict[k] = _smo_lst

    smo_res = []
    for i in range(N):
        x_s_info = {k: smo_dict[k][i] for k in keys}
        smo_res.append(x_s_info)
    return smo_res


class AvatarRegistrar:
    """
    source image|video -> rgb_list -> source_info
    """
    def __init__(
        self,
        insightface_det_cfg,
        landmark106_cfg,
        landmark203_cfg,
        landmark478_cfg,
        appearance_extractor_cfg,
        motion_extractor_cfg,
    ):
        self.source2info = Source2Info(
            insightface_det_cfg,
            landmark106_cfg,
            landmark203_cfg,
            landmark478_cfg,
            appearance_extractor_cfg,
            motion_extractor_cfg,
        )

    def close(self):
        self.source2info.close()

    def register(
        self,
        source_path,  # image | video
        max_dim=1920,
        n_frames=-1,
        **kwargs,
    ):
        """
        kwargs:
            crop_scale: 2.3
            crop_vx_ratio: 0
            crop_vy_ratio: -0.125
            crop_flag_do_rot: True
        """
        rgb_list, is_image_flag = load_source_frames(source_path, max_dim=max_dim, n_frames=n_frames)
        source_info = {
            "x_s_info_lst": [],
            "f_s_lst": [],
            "M_c2o_lst": [],
            "eye_open_lst": [],
            "eye_ball_lst": [],
        }
        keys = ["x_s_info", "f_s", "M_c2o", "eye_open", "eye_ball"]
        last_lmk = None
        for rgb in rgb_list:
            info = self.source2info(rgb, last_lmk, **kwargs)
            for k in keys:
                source_info[f"{k}_lst"].append(info[k])

            last_lmk = info["lmk203"]

        sc_f0 = source_info['x_s_info_lst'][0]['kp'].flatten()

        source_info["sc"] = sc_f0
        source_info["is_image_flag"] = is_image_flag
        source_info["img_rgb_lst"] = rgb_list

        return source_info
    
    def __call__(self, *args, **kwargs):
        return self.register(*args, **kwargs)
    
