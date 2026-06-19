import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt


def get_transforms(img_size=224):
    mri_mean = (0.485, 0.456, 0.406)
    mri_std = (0.229, 0.224, 0.225)

    train_transform = A.Compose([
        A.Resize(img_size, img_size),

        A.MedianBlur(blur_limit=7, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),

        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.95, 1.05),
            rotate=(-15, 15),
            shear=0,
            p=0.5
        ),

        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.Normalize(mean=mri_mean, std=mri_std),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.MedianBlur(blur_limit=7, p=1.0),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=mri_mean, std=mri_std),
        ToTensorV2()
    ])

    return train_transform, val_transform


# Data Augumnetation viusalization
def show_augmentation_effects(data_dir):
    train_tf, _ = get_transforms()
    cls = os.listdir(data_dir)[0]

    img_path = os.path.join(data_dir,
                            cls,
                            os.listdir(os.path.join(data_dir, cls))[0])
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    augmented = train_tf(image=img)["image"].permute(1,2,0).numpy()
    augmented = (augmented - augmented.min()) / (augmented.max() - augmented.min())

    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("Original")

    plt.subplot(1,2,2)
    plt.imshow(augmented)
    plt.title("Augmented ")

    plt.show()
