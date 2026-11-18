"""Nạp CLIP một chỗ duy nhất, dùng chung cho API (main.py) và script ingest.

Tự chọn thiết bị: NVIDIA GPU (cuda) -> Apple Silicon (mps) -> CPU.
Có thể ép bằng biến môi trường DEVICE=cpu|cuda|mps.
"""


def pick_device(preference: str = "auto") -> str:
    import torch

    if preference and preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class Clip:
    def __init__(self, model_name: str, pretrained: str, device_preference: str = "auto"):
        import open_clip

        self.device = pick_device(device_preference)
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def encode_texts(self, texts):
        """list[str] -> tensor (N, 512), đã chuẩn hóa độ dài về 1."""
        import torch

        with torch.no_grad():
            tokens = self.tokenizer(texts).to(self.device)
            vecs = self.model.encode_text(tokens)
            return torch.nn.functional.normalize(vecs.float(), dim=-1)

    def encode_images(self, pil_images):
        """list[PIL.Image] -> tensor (N, 512), đã chuẩn hóa độ dài về 1."""
        import torch

        with torch.no_grad():
            batch = torch.stack([self.preprocess(im) for im in pil_images]).to(self.device)
            vecs = self.model.encode_image(batch)
            return torch.nn.functional.normalize(vecs.float(), dim=-1)


def load_clip(model_name: str, pretrained: str, device_preference: str = "auto") -> Clip:
    return Clip(model_name, pretrained, device_preference)
