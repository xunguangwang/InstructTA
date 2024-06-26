import os
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset


def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())


class ImageFolderWithPaths(Dataset):
    def __init__(self, img_folder, transform=T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.Lambda(lambda img: img.convert("RGB")),
            T.Lambda(lambda img: to_tensor(img))])
        ):
        img_names = sorted(os.listdir(img_folder))
        self.img_paths = [os.path.join(img_folder, name) for name in img_names]

        self.transform = transform
    
    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.img_paths[index]
    
    def __len__(self):
        return len(self.img_paths)


class Caption(Dataset):
    def __init__(self, file_path):
        with open(os.path.join(file_path), 'r') as f:
            self.captions = [line.strip() for line in f.readlines()]
            f.close()

    def __getitem__(self, index):
        # cap = torch.from_numpy(self.captions[index])
        cap = self.captions[index]
        return cap

    def __len__(self):
        return len(self.captions)


if __name__ == "__main__":
    a = ImageFolderWithPaths('../data/ImageNet-1K/10K')
