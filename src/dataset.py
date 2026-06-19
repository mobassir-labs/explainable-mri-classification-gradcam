import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset

from .transforms import get_transforms


#Custom Dataset Class
class MRIDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.paths = []
        self.labels = []
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_path = os.path.join(data_dir, cls)
            if not os.path.isdir(cls_path): continue
            for img in os.listdir(cls_path):
                self.paths.append(os.path.join(cls_path, img))
                self.labels.append(self.class_to_idx[cls])
        self.transform = transform

    def crop_brain_contour(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            image = image[y:y+h, x:x+w]
        return image

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.crop_brain_contour(img)


        if self.transform:
            img = self.transform(image=img)["image"]

        return img, self.labels[idx]

    def __len__(self):
        return len(self.paths)


#DataLoader And Split
class TransformSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
          x = self.transform(image=x)["image"] if isinstance(x, np.ndarray) else x
        return x, y

    def __len__(self):
        return len(self.subset)

def get_loaders_with_val(train_dir,
                               test_dir,
                                     batch_size=64,
                                         val_split=0.2):
    train_tf, val_tf = get_transforms()


    base_dataset = MRIDataset(train_dir, transform=None)

    indices = list(range(len(base_dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)

    split = int(len(base_dataset) * val_split)
    train_idx, val_idx = indices[split:], indices[:split]

    train_ds = TransformSubset(Subset(base_dataset, train_idx), transform=train_tf)
    val_ds = TransformSubset(Subset(base_dataset, val_idx), transform=val_tf)
    test_ds = MRIDataset(test_dir, transform=val_tf)

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            shuffle=False)
    test_loader = DataLoader(test_ds,
                             batch_size=batch_size,
                             shuffle=False)

    return train_loader, val_loader, test_loader, base_dataset.classes
