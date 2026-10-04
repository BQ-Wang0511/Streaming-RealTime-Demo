import torch
import torch.nn as nn


class FlowMatchingProcess(nn.Module):
    """Inference-only Flow Matching sampler."""

    def __init__(
        self,
        model,
        time_steps=1000,
        sampling_steps=1,
        guidance_scale=1.3,
        predict_clean_motion=True,
    ):
        super().__init__()
        self.model = model
        self.time_steps = int(time_steps)
        self.sampling_steps = int(sampling_steps)
        self.guidance_scale = float(guidance_scale)
        self.predict_clean_motion = bool(predict_clean_motion)

    @staticmethod
    def _expand_time(values, shape):
        return values.view(-1, *((1,) * (len(shape) - 1)))

    def _to_velocity(self, noisy_motion, model_output, normalized_time):
        if not self.predict_clean_motion:
            return model_output
        time = self._expand_time(normalized_time, noisy_motion.shape).clamp_min(1e-3)
        return (noisy_motion - model_output) / time

    @torch.no_grad()
    def sample(
        self,
        shape,
        initial_motion,
        audio_features,
        habit_one_hot,
        noise=None,
    ):
        batch_size = shape[0]
        device = next(self.model.parameters()).device
        steps = max(1, min(self.sampling_steps, self.time_steps))
        times = torch.linspace(1.0, 0.0, steps=steps + 1, device=device)

        motion = noise.to(device) if noise is not None else torch.randn(shape, device=device)
        initial_motion = initial_motion.to(device)
        audio_features = audio_features.to(device)
        habit_one_hot = habit_one_hot.to(device)

        for current_time, next_time in zip(times[:-1], times[1:]):
            model_time = torch.full(
                (batch_size,),
                float(current_time * max(self.time_steps - 1, 1)),
                device=device,
                dtype=motion.dtype,
            )
            normalized_time = torch.full(
                (batch_size,),
                float(current_time),
                device=device,
                dtype=motion.dtype,
            )
            model_output = self.model.guided_forward(
                motion,
                initial_motion,
                audio_features,
                habit_one_hot,
                model_time,
                self.guidance_scale,
            )
            velocity = self._to_velocity(motion, model_output, normalized_time)
            motion = motion + float(next_time - current_time) * velocity

        return motion
