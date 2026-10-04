# Latent Motion Diffusion Model
import torch
import torch.nn as nn
from .lmdm_modules.model import MotionDecoder
from .lmdm_modules.utils import extract, make_beta_schedule


class LMDM(nn.Module):
    def __init__(
        self,
        motion_feat_dim=265,
        audio_feat_dim=1024+35,
        seq_frames=80,
        checkpoint='',
        device='cuda',
        clip_denoised=False,    # clip denoised (-1,1)
        multi_cond_frame=False,
    ):
        super().__init__()

        self.motion_feat_dim = motion_feat_dim
        self.audio_feat_dim = audio_feat_dim
        self.seq_frames = seq_frames
        self.device = device

        self.n_timestep = 1000
        self.clip_denoised = clip_denoised
        self.guidance_weight = 2

        self.model = MotionDecoder(
            nfeats=motion_feat_dim,
            seq_len=seq_frames,
            latent_dim=512,
            ff_size=1024,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
            cond_feature_dim=audio_feat_dim,
            multi_cond_frame=multi_cond_frame,
        )

        self.init_diff()

        self.sampling_timesteps = None

    def init_diff(self):
        n_timestep = self.n_timestep
        betas = torch.Tensor(
            make_beta_schedule(schedule="cosine", n_timestep=n_timestep)
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )
        self.register_buffer("sqrt_recip1m_alphas_cumprod", torch.sqrt(1.0 / (1.0 - alphas_cumprod)))

    def predict_noise_from_start(self, x_t, t, x0):
        a = extract(self.sqrt_recip1m_alphas_cumprod, t, x_t.shape)
        b = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return (a * x_t - x0 / b)
    
    def maybe_clip(self, x):
        if self.clip_denoised:
            return torch.clamp(x, min=-1., max=1.)
        else:
            return x
        
    def model_predictions(self, x, cond_frame, cond, t):
        weight = self.guidance_weight
        x_start = self.model.guided_forward(x, cond_frame, cond, t, weight)
        x_start = self.maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start
    
    @torch.no_grad()
    def forward(self, x, cond_frame, cond, time_cond):
        pred_noise, x_start = self.model_predictions(x, cond_frame, cond, time_cond)
        return pred_noise, x_start
    
    def load_model(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.eval()
        return self

    def setup(self, sampling_timesteps=50):
        if self.sampling_timesteps == sampling_timesteps:
            return
        
        self.sampling_timesteps = sampling_timesteps

        total_timesteps = self.n_timestep
        device = self.device
        eta = 1
        shape = (1, self.seq_frames, self.motion_feat_dim)

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        self.time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        self.time_cond_list = []
        self.alpha_next_sqrt_list = []
        self.sigma_list = []
        self.c_list = []
        self.noise_list = []

        for time, time_next in self.time_pairs:
            time_cond = torch.full((1,), time, device=device, dtype=torch.long)
            self.time_cond_list.append(time_cond)
            if time_next < 0:
                continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn(shape, device=device)

            self.alpha_next_sqrt_list.append(alpha_next.sqrt())
            self.sigma_list.append(sigma)
            self.c_list.append(c)
            self.noise_list.append(noise)

    @torch.no_grad()
    def ddim_sample(self, kp_cond, aud_cond, sampling_timesteps):
        self.setup(sampling_timesteps)

        cond_frame = kp_cond
        cond = aud_cond

        shape = (1, self.seq_frames, self.motion_feat_dim)
        x = torch.randn(shape, device=self.device)

        x_start = None
        i = 0
        for _, time_next in self.time_pairs:
            time_cond = self.time_cond_list[i]
            pred_noise, x_start = self.model_predictions(x, cond_frame, cond, time_cond)
            if time_next < 0:
                x = x_start
                continue

            alpha_next_sqrt = self.alpha_next_sqrt_list[i]
            c = self.c_list[i]
            sigma = self.sigma_list[i]
            noise = self.noise_list[i]
            x = x_start * alpha_next_sqrt + c * pred_noise + sigma * noise

            i += 1
        return x  # pred_kp_seq

