
import onnxruntime


class HubertStreamingONNX:
    def __init__(self, model_file, device="cuda"):
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        
        self.session = onnxruntime.InferenceSession(model_file, providers=providers)

    def forward_chunk(self, input_values):
        encoding_out = self.session.run(
            None,
            {"input_values": input_values.reshape(1, -1)}
        )[0]
        return encoding_out
    

