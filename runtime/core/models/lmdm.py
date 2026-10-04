import torch

from ..utils.load_model import load_model


class LMDM:
    def __init__(self, model_path, device="cuda", **kwargs):
        kwargs["module_name"] = "LMDM"
        self.model = load_model(model_path, device=device, **kwargs)
        self.device = device
        self.motion_feat_dim = kwargs.get("motion_feat_dim", 265)
        self.audio_feat_dim = kwargs.get("audio_feat_dim", 1103)
        self.seq_frames = kwargs.get("seq_frames", 80)

    def setup(self, sampling_timesteps):
        self.model.setup(sampling_timesteps)

    def __call__(self, kp_cond, aud_cond, sampling_timesteps):
        return (
            self.model.ddim_sample(
                torch.from_numpy(kp_cond).to(self.device),
                torch.from_numpy(aud_cond).to(self.device),
                sampling_timesteps,
            )
            .cpu()
            .numpy()
        )
