from typing import Dict, List, Any
from PIL import Image

import os
import json
import torch
import torchvision
from torch.nn import functional as F

from .midjourney5M import Model5M

class PreTrainedPipeline():
    def __init__(self, path=""):
        self.model = Model5M()
        ckpt = torch.load(os.path.join(path, "midjourney5M.pt"), map_location=torch.device('cpu'))
        self.model.load_state_dict(ckpt)
        self.model.eval()

        with open(os.path.join(path, "config.json")) as config:
            config = json.load(config)
        self.id2label = config["id2label"]
        
        self.tfm = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((224, 224)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                ])

    def __call__(self, inputs: "Image.Image") -> List[Dict[str, Any]]:
        """
        Args:
            inputs (:obj:`PIL.Image`):
                The raw image representation as PIL.
                No transformation made whatsoever from the input. Make all necessary transformations here.
        Return:
            A :obj:`list`:. The list contains items that are dicts should be liked {"label": "XXX", "score": 0.82}
                It is preferred if the returned list is in decreasing `score` order
        """
        img = self.tfm(inputs)
        return self.predict_from_model(img)

    def predict_from_model(self, img):
        y = self.model.forward(img[None, ...])
        y_1 = F.softmax(y, dim=1)[:, 1].cpu().detach().numpy()
        y_2 = F.softmax(y, dim=1)[:, 0].cpu().detach().numpy()
        labels = [
            {"label": str(self.id2label["0"]), "score": y_1.tolist()[0]},
            {"label": str(self.id2label["1"]), "score": y_2.tolist()[0]},
        ]
        return labels
