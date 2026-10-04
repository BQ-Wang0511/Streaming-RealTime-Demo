import copy
import random
import numpy as np
from scipy.special import softmax

from ..models.stitch_network import StitchNetwork


"""
# __init__
stitch_network_cfg = {
    "model_path": "",
    "device": "cuda",
}

# __call__
kwargs:
    fade_alpha
    fade_out_keys

    delta_pitch
    delta_yaw
    delta_roll

"""


def ctrl_motion(x_d_info, **kwargs):
    # pose + offset
    for kk in ["delta_pitch", "delta_yaw", "delta_roll"]:
        if kk in kwargs:
            k = kk[6:]
            x_d_info[k] = bin66_to_degree(x_d_info[k]) + kwargs[kk]

    # pose * alpha
    for kk in ["alpha_pitch", "alpha_yaw", "alpha_roll"]:
        if kk in kwargs:
            k = kk[6:]
            x_d_info[k] = x_d_info[k] * kwargs[kk]

    # exp + offset
    if "delta_exp" in kwargs:
        k = "exp"
        x_d_info[k] = x_d_info[k] + kwargs["delta_exp"]

    return x_d_info


def fade(x_d_info, dst, alpha, keys=None):
    if keys is None:
        keys = x_d_info.keys()
    for k in keys:
        if k == 'kp':
            continue
        x_d_info[k] = x_d_info[k] * alpha + dst[k] * (1 - alpha)
    return x_d_info


def ctrl_vad(x_d_info, dst, alpha):
    exp = x_d_info["exp"]
    exp_dst = dst["exp"]

    _lip = [6, 12, 14, 17, 19, 20]
    _a1 = np.zeros((21, 3), dtype=np.float32)
    _a1[_lip] = alpha
    _a1 = _a1.reshape(1, -1)
    x_d_info["exp"] = exp * alpha + exp_dst * (1 - alpha)

    return x_d_info
    
    

def _mix_s_d_info(
    x_s_info,
    x_d_info,
    use_d_keys=("exp", "pitch", "yaw", "roll", "t"),
    d0=None,
):
    if d0 is not None:
        if isinstance(use_d_keys, dict):
            x_d_info = {
                k: x_s_info[k] + (v - d0[k]) * use_d_keys.get(k, 1)
                for k, v in x_d_info.items()
            }
        else:
            x_d_info = {k: x_s_info[k] + (v - d0[k]) for k, v in x_d_info.items()}

    for k, v in x_s_info.items():
        if k not in x_d_info or k not in use_d_keys:
            x_d_info[k] = v

    if isinstance(use_d_keys, dict) and d0 is None:
        for k, alpha in use_d_keys.items():
            x_d_info[k] *= alpha
    return x_d_info


def _set_eye_blink_idx(N, blink_n=15, open_n=-1):
    """
    open_n:
        -1: no blink
        0: random open_n
        >0: fix open_n
        list: loop open_n
    """
    OPEN_MIN = 60
    OPEN_MAX = 100

    idx = [0] * N
    if isinstance(open_n, int):
        if open_n < 0:  # no blink
            return idx
        elif open_n > 0:  # fix open_n
            open_ns = [open_n]
        else:  # open_n == 0:  # random open_n, 60-100
            open_ns = []
    elif isinstance(open_n, list):
        open_ns = open_n  # loop open_n
    else:
        raise ValueError()

    blink_idx = list(range(blink_n))

    start_n = open_ns[0] if open_ns else random.randint(OPEN_MIN, OPEN_MAX)
    end_n = open_ns[-1] if open_ns else random.randint(OPEN_MIN, OPEN_MAX)
    max_i = N - max(end_n, blink_n)
    cur_i = start_n
    cur_n_i = 1
    while cur_i < max_i:
        idx[cur_i : cur_i + blink_n] = blink_idx

        if open_ns:
            cur_n = open_ns[cur_n_i % len(open_ns)]
            cur_n_i += 1
        else:
            cur_n = random.randint(OPEN_MIN, OPEN_MAX)

        cur_i = cur_i + blink_n + cur_n

    return idx


def _fix_exp_for_x_d_info(x_d_info, x_s_info, delta_eye=None, drive_eye=True):
    _eye = [11, 13, 15, 16, 18]
    _lip = [6, 12, 14, 17, 19, 20]
    alpha = np.zeros((21, 3), dtype=x_d_info["exp"].dtype)
    alpha[_lip] = 1
    if delta_eye is None and drive_eye:  # use d eye
        alpha[_eye] = 1
    alpha = alpha.reshape(1, -1)
    x_d_info["exp"] = x_d_info["exp"] * alpha + x_s_info["exp"] * (1 - alpha)

    if delta_eye is not None and drive_eye:
        alpha = np.zeros((21, 3), dtype=x_d_info["exp"].dtype)
        alpha[_eye] = 1
        alpha = alpha.reshape(1, -1)
        x_d_info["exp"] = (delta_eye + x_s_info["exp"]) * alpha + x_d_info["exp"] * (
            1 - alpha
        )

    return x_d_info


def _fix_exp_for_x_d_info_v2(x_d_info, x_s_info, delta_eye, a1, a2, a3):
    x_d_info["exp"] = x_d_info["exp"] * a1 + x_s_info["exp"] * a2 + delta_eye * a3
    return x_d_info


def bin66_to_degree(pred):
    if pred.ndim > 1 and pred.shape[1] == 66:
        idx = np.arange(66).astype(np.float32)
        pred = softmax(pred, axis=1)
        degree = np.sum(pred * idx, axis=1) * 3 - 97.5
        return degree
    return pred


def _eye_delta(exp, dx=0, dy=0):
    if dx > 0:
        exp[0, 33] += dx * 0.0007
        exp[0, 45] += dx * 0.001
    else:
        exp[0, 33] += dx * 0.001
        exp[0, 45] += dx * 0.0007

    exp[0, 34] += dy * -0.001
    exp[0, 46] += dy * -0.001
    return exp

def _fix_gaze(pose_s, x_d_info):
    x_ratio = 0.26
    y_ratio = 0.28
    
    yaw_s, pitch_s = pose_s
    yaw_d = bin66_to_degree(x_d_info['yaw']).item()
    pitch_d = bin66_to_degree(x_d_info['pitch']).item()

    delta_yaw = yaw_d - yaw_s
    delta_pitch = pitch_d - pitch_s

    dx = delta_yaw * x_ratio
    dy = delta_pitch * y_ratio
    
    x_d_info['exp'] = _eye_delta(x_d_info['exp'], dx, dy)
    return x_d_info


def get_rotation_matrix(pitch_, yaw_, roll_):
    """ the input is in degree
    """
    # transform to radian
    pitch = pitch_ / 180 * np.pi
    yaw = yaw_ / 180 * np.pi
    roll = roll_ / 180 * np.pi

    if pitch.ndim == 1:
        pitch = pitch[:, None]
    if yaw.ndim == 1:
        yaw = yaw[:, None]
    if roll.ndim == 1:
        roll = roll[:, None]

    # calculate the euler matrix
    bs = pitch.shape[0]
    ones = np.ones((bs, 1), dtype=np.float32)
    zeros = np.zeros((bs, 1), dtype=np.float32)
    x, y, z = pitch, yaw, roll

    rot_x = np.concatenate([
        ones, zeros, zeros,
        zeros, np.cos(x), -np.sin(x),
        zeros, np.sin(x), np.cos(x)
    ], axis=1).reshape(bs, 3, 3)

    rot_y = np.concatenate([
        np.cos(y), zeros, np.sin(y),
        zeros, ones, zeros,
        -np.sin(y), zeros, np.cos(y)
    ], axis=1).reshape(bs, 3, 3)

    rot_z = np.concatenate([
        np.cos(z), -np.sin(z), zeros,
        np.sin(z), np.cos(z), zeros,
        zeros, zeros, ones
    ], axis=1).reshape(bs, 3, 3)

    rot = np.matmul(np.matmul(rot_z, rot_y), rot_x)
    return np.transpose(rot, (0, 2, 1))


def transform_keypoint(kp_info: dict):
    """
    transform the implicit keypoints with the pose, shift, and expression deformation
    kp: BxNx3
    """
    kp = kp_info['kp']    # (bs, k, 3)
    pitch, yaw, roll = kp_info['pitch'], kp_info['yaw'], kp_info['roll']

    t, exp = kp_info['t'], kp_info['exp']
    scale = kp_info['scale']

    pitch = bin66_to_degree(pitch)
    yaw = bin66_to_degree(yaw)
    roll = bin66_to_degree(roll)

    bs = kp.shape[0]
    if kp.ndim == 2:
        num_kp = kp.shape[1] // 3  # Bx(num_kpx3)
    else:
        num_kp = kp.shape[1]  # Bxnum_kpx3

    rot_mat = get_rotation_matrix(pitch, yaw, roll)    # (bs, 3, 3)

    # Eqn.2: s * (R * x_c,s + exp) + t
    kp_transformed = np.matmul(kp.reshape(bs, num_kp, 3), rot_mat) + exp.reshape(bs, num_kp, 3)
    kp_transformed *= scale[..., None]  # (bs, k, 3) * (bs, 1, 1) = (bs, k, 3)
    kp_transformed[:, :, 0:2] += t[:, None, 0:2]  # remove z, only apply tx ty

    return kp_transformed


class MotionStitch:
    def __init__(
        self,
        stitch_network_cfg,
    ):
        self.stitch_net = StitchNetwork(**stitch_network_cfg)

    def set_Nd(self, N_d=-1):
        # only for offline (make start|end eye open)
        if N_d == self.N_d:
            return
        
        self.N_d = N_d
        if self.drive_eye and self.delta_eye_arr is not None:
            N = 3000 if self.N_d == -1 else self.N_d
            self.delta_eye_idx_list = _set_eye_blink_idx(
                N, len(self.delta_eye_arr), self.delta_eye_open_n
            )

    def setup(
        self,
        N_d=-1,
        use_d_keys=None,
        relative_d=True,
        drive_eye=None,  # use d eye or s eye
        delta_eye_arr=None,  # fix eye
        delta_eye_open_n=-1,  # int|list
        fade_out_keys=("exp",),
        fade_type="",   # "" | "d0" | "s"
        flag_stitching=True,
        is_image_flag=True,
        x_s_info=None,
        d0=None,
        ch_info=None,
        overall_ctrl_info=None,
    ):
        self.is_image_flag = is_image_flag
        if use_d_keys is None:
            if self.is_image_flag:
                self.use_d_keys = ("exp", "pitch", "yaw", "roll", "t")
            else:
                self.use_d_keys = ("exp", )
        else:
            self.use_d_keys = use_d_keys

        if drive_eye is None:
            if self.is_image_flag:
                self.drive_eye = True
            else:
                self.drive_eye = False
        else:
            self.drive_eye = drive_eye

        self.N_d = N_d
        self.relative_d = relative_d
        self.delta_eye_arr = delta_eye_arr
        self.delta_eye_open_n = delta_eye_open_n
        self.fade_out_keys = fade_out_keys
        self.fade_type = fade_type
        self.flag_stitching = flag_stitching

        _eye = [11, 13, 15, 16, 18]
        _lip = [6, 12, 14, 17, 19, 20]
        _a1 = np.zeros((21, 3), dtype=np.float32)
        _a1[_lip] = 1
        _a2 = 0
        if self.drive_eye:
            if self.delta_eye_arr is None:
                _a1[_eye] = 1
            else:
                _a2 = np.zeros((21, 3), dtype=np.float32)
                _a2[_eye] = 1
                _a2 = _a2.reshape(1, -1)
        _a1 = _a1.reshape(1, -1)

        self.fix_exp_a1 = _a1 * (1 - _a2)
        self.fix_exp_a2 = (1 - _a1) + _a1 * _a2
        self.fix_exp_a3 = _a2

        if self.drive_eye and self.delta_eye_arr is not None:
            N = 3000 if self.N_d == -1 else self.N_d
            self.delta_eye_idx_list = _set_eye_blink_idx(
                N, len(self.delta_eye_arr), self.delta_eye_open_n
            )

        self.pose_s = None
        self.x_s = None
        self.fade_dst = None
        if self.is_image_flag and x_s_info is not None:
            yaw_s = bin66_to_degree(x_s_info['yaw']).item()
            pitch_s = bin66_to_degree(x_s_info['pitch']).item()
            self.pose_s = [yaw_s, pitch_s]
            self.x_s = transform_keypoint(x_s_info)

            if self.fade_type == "s":
                self.fade_dst = copy.deepcopy(x_s_info)

        if ch_info is not None:
            self.scale_a = ch_info['x_s_info_lst'][0]['scale'].item()
            if x_s_info is not None:
                self.scale_b = x_s_info['scale'].item()
                self.scale_ratio = self.scale_a / self.scale_b
                self._set_scale_ratio(self.scale_ratio)
            else:
                self.scale_ratio = None
        else:
            self.scale_ratio = 1

        self.overall_ctrl_info = overall_ctrl_info

        self.d0 = d0
        self.idx = 0

    def _set_scale_ratio(self, scale_ratio=1):
        if scale_ratio == 1:
            return
        if isinstance(self.use_d_keys, dict):
            self.use_d_keys = {k: v * (scale_ratio if k in {"exp", "pitch", "yaw", "roll"} else 1) for k, v in self.use_d_keys.items()}
        else:
            self.use_d_keys = {k: scale_ratio if k in {"exp", "pitch", "yaw", "roll"} else 1 for k in self.use_d_keys}

    @staticmethod
    def _merge_kwargs(default_kwargs, run_kwargs):
        if default_kwargs is None:
            return run_kwargs
        
        for k, v in default_kwargs.items():
            if k not in run_kwargs:
                run_kwargs[k] = v
        return run_kwargs
    
    def __call__(self, x_s_info, x_d_info, **kwargs):

        kwargs = self._merge_kwargs(self.overall_ctrl_info, kwargs)

        if self.scale_ratio is None:
            self.scale_b = x_s_info['scale'].item()
            self.scale_ratio = self.scale_a / self.scale_b
            self._set_scale_ratio(self.scale_ratio)

        if self.relative_d and self.d0 is None:
            self.d0 = copy.deepcopy(x_d_info)

        x_d_info = _mix_s_d_info(
            x_s_info,
            x_d_info,
            self.use_d_keys,
            self.d0,
        )

        delta_eye = 0
        if self.drive_eye and self.delta_eye_arr is not None:
            delta_eye = self.delta_eye_arr[
                self.delta_eye_idx_list[self.idx % len(self.delta_eye_idx_list)]
            ][None]
        x_d_info = _fix_exp_for_x_d_info_v2(
            x_d_info,
            x_s_info,
            delta_eye,
            self.fix_exp_a1,
            self.fix_exp_a2,
            self.fix_exp_a3,
        )

        if kwargs.get("vad_alpha", 1) < 1:
            x_d_info = ctrl_vad(x_d_info, x_s_info, kwargs.get("vad_alpha", 1))

        x_d_info = ctrl_motion(x_d_info, **kwargs)

        if self.fade_type == "d0" and self.fade_dst is None:
            self.fade_dst = copy.deepcopy(x_d_info)

        # fade
        if "fade_alpha" in kwargs and self.fade_type in ["d0", "s"]:
            fade_alpha = kwargs["fade_alpha"]
            fade_keys = kwargs.get("fade_out_keys", self.fade_out_keys)
            if self.fade_type == "d0":
                fade_dst = self.fade_dst
            elif self.fade_type == "s":
                if self.fade_dst is not None:
                    fade_dst = self.fade_dst
                else:
                    fade_dst = copy.deepcopy(x_s_info)
                    if self.is_image_flag:
                        self.fade_dst = fade_dst
            x_d_info = fade(x_d_info, fade_dst, fade_alpha, fade_keys)

        if self.drive_eye:
            if self.pose_s is None:
                yaw_s = bin66_to_degree(x_s_info['yaw']).item()
                pitch_s = bin66_to_degree(x_s_info['pitch']).item()
                self.pose_s = [yaw_s, pitch_s]
            x_d_info = _fix_gaze(self.pose_s, x_d_info)

        if self.x_s is not None:
            x_s = self.x_s
        else:
            x_s = transform_keypoint(x_s_info)
            if self.is_image_flag:
                self.x_s = x_s
        
        x_d = transform_keypoint(x_d_info)
        
        if self.flag_stitching:
            x_d = self.stitch_net(x_s, x_d)

        self.idx += 1

        return x_s, x_d
