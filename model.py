import sys
import torch

sys.path.insert(0, "third_party/aasist")

from models.AASIST import Model


class VoiceCloneModel:
    def __init__(self):
        config = {
            "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
            "gat_dims": [64, 32],
            "pool_ratios": [0.5, 0.7, 0.5, 0.5],
            "temperatures": [2.0, 2.0, 100.0, 100.0],
            "first_conv": 128,
        }

        self.model = Model(config)

        weights_path = "third_party/aasist/models/weights/AASIST.pth"

        self.model.load_state_dict(
            torch.load(weights_path, map_location="cpu")
        )

        self.model.eval()

    def predict(self, audio):
        audio = torch.tensor(audio, dtype=torch.float32)

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        with torch.no_grad():
            _, output = self.model(audio)

        probabilities = torch.softmax(output, dim=1)

        spoof_probability = probabilities[0, 0].item()
        bonafide_probability = probabilities[0, 1].item()
        print("SPOOF SCORE:", spoof_probability)
        print("BONAFIDE SCORE:", bonafide_probability)

        if spoof_probability >= bonafide_probability:
            return {
                "label": "FAKE / SPOOF",
                "confidence": round(spoof_probability * 100, 2)
            }

        return {
            "label": "REAL / BONAFIDE",
            "confidence": round(bonafide_probability * 100, 2)
        }