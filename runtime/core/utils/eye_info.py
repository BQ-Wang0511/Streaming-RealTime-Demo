import numpy as np
from dataclasses import dataclass


@dataclass
class EyeIdxMP:
    LO = [33]
    LI = [133]
    LD = [7, 163, 144, 145, 153, 154, 155]  # O -> I
    LU = [246, 161, 160, 159, 158, 157, 173]  # O -> I
    RO = [263]
    RI = [362]
    RD = [249, 390, 373, 374, 380, 381, 382]  # O -> I
    RU = [466, 388, 387, 386, 385, 384, 398]  # O -> I

    LW = [33, 133]    # oi
    LH0 = [145, 159]
    LH1 = [144, 160]
    LH2 = [153, 158]

    RW = [263, 362]   # oi
    RH0 = [374, 386]
    RH1 = [373, 387]
    RH2 = [380, 385]

    LB = [468]  # eye ball
    RB = [473]


class EyeAttrUtilsByMP:
    def __init__(self, lmks_mp):
        self.IDX = EyeIdxMP()
        self.lmks = lmks_mp    # [n, 478, 3]

        self.L_width = self._dist_idx(*self.IDX.LW)   # [n]
        self.R_width = self._dist_idx(*self.IDX.RW)

        self.L_h0 = self._dist_idx(*self.IDX.LH0)
        self.L_h1 = self._dist_idx(*self.IDX.LH1)
        self.L_h2 = self._dist_idx(*self.IDX.LH2)

        self.R_h0 = self._dist_idx(*self.IDX.RH0)
        self.R_h1 = self._dist_idx(*self.IDX.RH1)
        self.R_h2 = self._dist_idx(*self.IDX.RH2)

        self.L_open =  (self.L_h0 + self.L_h1 + self.L_h2) / (self.L_width + 1e-8)   # [n]
        self.R_open =  (self.R_h0 + self.R_h1 + self.R_h2) / (self.R_width + 1e-8)

        self.L_center = self._center_idx(*self.IDX.LW)    # [n, 3/2]
        self.R_center = self._center_idx(*self.IDX.RW)

        self.L_ball = self.lmks[:, self.IDX.LB[0]]   # [n, 3/2]
        self.R_ball = self.lmks[:, self.IDX.RB[0]]

        self.L_ball_direc = (self.L_ball - self.L_center) / (self.L_width[:, None] + 1e-8)   # [n, 3/2]
        self.R_ball_direc = (self.R_ball - self.R_center) / (self.R_width[:, None] + 1e-8)

        self.L_eye_direc = self._direc_idx(*self.IDX.LW)  # I->O
        self.R_eye_direc = self._direc_idx(*self.IDX.RW)

        self.L_ball_move_dist = self._dist(self.L_ball, self.L_center)
        self.R_ball_move_dist = self._dist(self.R_ball, self.R_center)

        self.L_ball_move_direc = self._direc(self.L_ball, self.L_center) - self.L_eye_direc
        self.R_ball_move_direc = self._direc(self.R_ball, self.R_center) - self.R_eye_direc

        self.L_ball_move = self.L_ball_move_direc * self.L_ball_move_dist[:, None]
        self.R_ball_move = self.R_ball_move_direc * self.R_ball_move_dist[:, None]

    def LR_open(self):
        LR_open = np.stack([self.L_open, self.R_open], -1)    # [n, 2]
        return LR_open
    
    def LR_ball_direc(self):
        LR_ball_direc = np.stack([self.L_ball_direc, self.R_ball_direc], -1)    # [n, 3, 2]
        return LR_ball_direc
    
    def LR_ball_move(self):
        LR_ball_move = np.stack([self.L_ball_move, self.R_ball_move], -1)
        return LR_ball_move

    @staticmethod
    def _dist(p1, p2):
        # p1/p2: [n, 3/2]
        return (((p1 - p2) ** 2).sum(-1)) ** 0.5    # [n]
    
    @staticmethod
    def _center(p1, p2):
        return (p1 + p2) * 0.5   # [n, 3/2]
    
    def _direc(self, p1, p2):
        # p1 - p2, (2->1)
        return (p1 - p2) / (self._dist(p1, p2)[:, None] + 1e-8)
    
    def _dist_idx(self, idx1, idx2):
        p1 = self.lmks[:, idx1]
        p2 = self.lmks[:, idx2]
        d = self._dist(p1, p2)
        return d
    
    def _center_idx(self, idx1, idx2):
        p1 = self.lmks[:, idx1]
        p2 = self.lmks[:, idx2]
        c = self._center(p1, p2)
        return c
    
    def _direc_idx(self, idx1, idx2):
        p1 = self.lmks[:, idx1]
        p2 = self.lmks[:, idx2]
        dir = self._direc(p1, p2)
        return dir