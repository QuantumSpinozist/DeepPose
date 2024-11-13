import deeplake
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import numpy as np

from PIL import Image

class LSPDataset(Dataset):
    def __init__(self, ds, transform = None):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        image = self.ds['images'][idx].numpy()
        image = Image.fromarray(image.astype('uint8'), 'RGB')  # Make sure it's uint8

        label = self.ds['keypoints'][idx].numpy(fetch_chunks = True)[:,:2].astype(np.float32)

        if self.transform is not None:
            image = self.transform(image)

        sample = (image, label)

        return sample