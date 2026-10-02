import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class PairedImageFolder(Dataset):
    """Raw images and their precomputed crack-aware views.

    `raw_root` and `aug_root` share the same <class>/<image> layout; each raw
    image is paired with the crack-aware view at the same relative path.
    """

    def __init__(self, raw_root: str, aug_root: str):
        folder = ImageFolder(raw_root)
        self.samples = folder.samples
        self.classes = folder.classes
        self.raw_root = raw_root
        self.aug_root = aug_root

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        aug_path = os.path.join(self.aug_root, os.path.relpath(path, self.raw_root))
        raw = transform(Image.open(path).convert('RGB'))
        aug = transform(Image.open(aug_path).convert('RGB'))
        return raw, aug, label


def image_folder(root: str) -> ImageFolder:
    return ImageFolder(root, transform=transform)
