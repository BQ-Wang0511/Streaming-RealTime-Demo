from .modules.hubert_stream import HubertStreamingONNX


class HubertStreaming:
    def __init__(self, model_path, device="cuda", **_):
        self.model = HubertStreamingONNX(model_file=model_path, device=device)

    def __call__(self, audio_chunk):
        return self.model.forward_chunk(audio_chunk)
