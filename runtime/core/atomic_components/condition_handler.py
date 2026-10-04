import numpy as np
from scipy.special import softmax
import copy


def _get_emo_avg(idx=6):
    emo_avg = np.zeros(8, dtype=np.float32)
    if isinstance(idx, (list, tuple)):
        for i in idx:
            emo_avg[i] = 8
    else:
        emo_avg[idx] = 8
    emo_avg = softmax(emo_avg)
    return emo_avg


def _mirror_index(index, size):
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    else:
        return size - res - 1
    

class ConditionHandler:
    """
    aud_feat, emo_seq, eye_seq, sc_seq -> cond_seq
    """
    def __init__(
        self,
        use_emo=True,
        use_sc=True,
        use_eye_open=True,
        use_eye_ball=True,
        seq_frames=80,
    ):
        self.use_emo = use_emo
        self.use_sc = use_sc
        self.use_eye_open = use_eye_open
        self.use_eye_ball = use_eye_ball

        self.seq_frames = seq_frames

    def setup(self, setup_info, emo, eye_f0_mode=False, ch_info=None):
        """
        emo: int | [int] | [[int]] | numpy
        """
        if ch_info is None:
            source_info = copy.deepcopy(setup_info)
        else:
            source_info = ch_info

        self.eye_f0_mode = eye_f0_mode
        self.x_s_info_0 = source_info['x_s_info_lst'][0]

        if self.use_sc:
            self.sc = source_info["sc"]    # 63
            self.sc_seq = np.stack([self.sc] * self.seq_frames, 0)
        
        if self.use_eye_open:
            self.eye_open_lst = np.concatenate(source_info["eye_open_lst"], 0)  # [n, 2]
            self.num_eye_open = len(self.eye_open_lst)
            if self.num_eye_open == 1 or self.eye_f0_mode:
                self.eye_open_seq = np.stack([self.eye_open_lst[0]] * self.seq_frames, 0)
            else:
                self.eye_open_seq = None
        
        if self.use_eye_ball:
            self.eye_ball_lst = np.concatenate(source_info["eye_ball_lst"], 0)  # [n, 6]
            self.num_eye_ball = len(self.eye_ball_lst)
            if self.num_eye_ball == 1 or self.eye_f0_mode:
                self.eye_ball_seq = np.stack([self.eye_ball_lst[0]] * self.seq_frames, 0)
            else:
                self.eye_ball_seq = None

        if self.use_emo:
            self.emo_lst = self._parse_emo_seq(emo)
            self.num_emo = len(self.emo_lst)
            if self.num_emo == 1:
                self.emo_seq = np.concatenate([self.emo_lst] * self.seq_frames, 0)
            else:
                self.emo_seq = None

    @staticmethod
    def _parse_emo_seq(emo, seq_len=-1):
        if isinstance(emo, np.ndarray) and emo.ndim == 2 and emo.shape[1] == 8:
            # emo arr, e.g. real
            emo_seq = emo   # [m, 8]
        elif isinstance(emo, int) and 0 <= emo < 8:
            # emo label, e.g. 4
            emo_seq = _get_emo_avg(emo).reshape(1, 8)    # [1, 8]
        elif isinstance(emo, (list, tuple)) and 0 < len(emo) < 8 and isinstance(emo[0], int):
            # emo labels, e.g. [3,4]
            emo_seq = _get_emo_avg(emo).reshape(1, 8)    # [1, 8]
        elif isinstance(emo, list) and emo and isinstance(emo[0], (list, tuple)):
            # emo label list, e.g. [[4], [3,4], [3],[3,4,5], ...]
            emo_seq = np.stack([_get_emo_avg(i) for i in emo], 0)    # [m, 8]
        else:
            raise ValueError(f"Unsupported emo type: {emo}")
    
        if seq_len > 0:
            if len(emo_seq) == seq_len:
                return emo_seq
            elif len(emo_seq) == 1:
                return np.concatenate([emo_seq] * seq_len, 0)
            elif len(emo_seq) > seq_len:
                return emo_seq[:seq_len]
            else:
                raise ValueError(f"emo len {len(emo_seq)} can not match seq len ({seq_len})")
        else:
            return emo_seq
        
    def __call__(self, aud_feat, idx, emo=None):
        """
        aud_feat: [n, 1024]
        idx: int, <0 means pad (first clip buffer)
        """

        frame_num = len(aud_feat)
        more_cond = [aud_feat]
        if self.use_emo:
            if emo is not None:
                emo_seq = self._parse_emo_seq(emo, frame_num)
            elif self.emo_seq is not None and len(self.emo_seq) == frame_num:
                emo_seq = self.emo_seq
            else:
                emo_idx_list = [max(i, 0) % self.num_emo for i in range(idx, idx + frame_num)]
                emo_seq = self.emo_lst[emo_idx_list]
            more_cond.append(emo_seq)

        if self.use_eye_open:
            if self.eye_open_seq is not None and len(self.eye_open_seq) == frame_num:
                eye_open_seq = self.eye_open_seq
            else:
                if self.eye_f0_mode:
                    eye_idx_list = [0] * frame_num
                else:
                    eye_idx_list = [_mirror_index(max(i, 0), self.num_eye_open) for i in range(idx, idx + frame_num)]
                eye_open_seq = self.eye_open_lst[eye_idx_list]
            more_cond.append(eye_open_seq)

        if self.use_eye_ball:
            if self.eye_ball_seq is not None and len(self.eye_ball_seq) == frame_num:
                eye_ball_seq = self.eye_ball_seq
            else:
                if self.eye_f0_mode:
                    eye_idx_list = [0] * frame_num
                else:
                    eye_idx_list = [_mirror_index(max(i, 0), self.num_eye_ball) for i in range(idx, idx + frame_num)]
                eye_ball_seq = self.eye_ball_lst[eye_idx_list]
            more_cond.append(eye_ball_seq)

        if self.use_sc:
            if len(self.sc_seq) == frame_num:
                sc_seq = self.sc_seq
            else:
                sc_seq = np.stack([self.sc] * frame_num, 0)
            more_cond.append(sc_seq)

        if len(more_cond) > 1:
            cond_seq = np.concatenate(more_cond, -1)    # [n, dim_cond]
        else:
            cond_seq = aud_feat

        return cond_seq
