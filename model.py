import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

class SimpleUNet:
    def __init__(self, checkpoint_path=None):
        # заглушка, можно заменить на полноценный U-Net или SegFormer
        self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', pretrained=True)
        self.model.eval()
        if checkpoint_path:
            self.model.load_state_dict(torch.load(checkpoint_path))

        self.transform = T.Compose([
            T.Resize((256,256)),
            T.ToTensor(),
        ])

    def predict(self, image_path):
        img = Image.open(image_path).convert('RGB')
        x = self.transform(img).unsqueeze(0)  # batch 1
        with torch.no_grad():
            y = self.model(x)
        mask = (y.squeeze().numpy() > 0.5).astype(np.uint8)
        return mask

