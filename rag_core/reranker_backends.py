"""CrossEncoder execution backends."""

import numpy as np


class LazyCrossEncoder:
    """Defer model allocation until pipeline warmup or the first real query."""

    def __init__(self, model_name, *, max_length=512, device="cuda"):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
                local_files_only=True,
            )
        return self._model

    def predict(self, pairs, show_progress_bar=False):
        return self._load().predict(
            pairs, show_progress_bar=show_progress_bar
        )


class TensorRTCrossEncoder:
    def __init__(self, model_name, onnx_path):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        providers = [
            ("TensorrtExecutionProvider", {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "./trt_cache",
                "trt_fp16_enable": True,
            }),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        self.session = ort.InferenceSession(onnx_path, providers=providers)

    def predict(self, pairs, show_progress_bar=False):
        inputs = self.tokenizer(
            pairs,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        ort_inputs = {key: value.astype(np.int64) for key, value in inputs.items()}
        return self.session.run(None, ort_inputs)[0].flatten()
