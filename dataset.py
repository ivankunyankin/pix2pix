import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from utils import *


class SimpsonsDataset(Dataset):
    def __init__(self, root_dir):
        super(SimpsonsDataset, self).__init__()
        self.root_dir = root_dir
        self.collection = []
        for image_path in os.listdir(root_dir):
            if image_path.endswith(".png"):
                contour_path = image_path.replace(".png", f"_contour.png")
                if os.path.exists(os.path.join(self.root_dir, contour_path)):
                    self.collection.append((
                        os.path.join(os.path.join(self.root_dir, image_path)),
                        os.path.join(os.path.join(self.root_dir, contour_path)),
                    ))

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, index):
        image_path, contour_path = self.collection[index]
        image = np.array(Image.open(image_path))
        contour = np.array(Image.open(contour_path))

        augmentations = both_transform(image=image, image0=contour)
        image = augmentations["image"]
        contour = augmentations["image0"]

        image = transform_only_input(image=image)["image"]
        contour = transform_only_mask(image=contour)["image"]

        return contour, image
