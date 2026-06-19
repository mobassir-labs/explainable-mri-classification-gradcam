import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def visualize_brain_tumors_localization(model, dataloader, classes, device='cuda'):
    model.eval()
    target_layers = [model.cnn_backbone.stages[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    mri_mean = np.array([0.485, 0.456, 0.406])
    mri_std = np.array([0.229, 0.224, 0.225])

    # 🔥 Step 1: Collect multiple samples per class
    class_samples = {i: [] for i in range(len(classes))}

    for images, labels in dataloader:
        for i in range(len(labels)):
            lbl = labels[i].item()
            class_samples[lbl].append(images[i])

    # 🔥 Step 2: Pick RANDOM one per class
    found_samples = {}
    for cls in class_samples:
        if len(class_samples[cls]) > 0:
            found_samples[cls] = random.choice(class_samples[cls])

    # Plot
    fig, axes = plt.subplots(2, len(classes), figsize=(20, 10))

    for idx, class_idx in enumerate(sorted(found_samples.keys())):
        img_tensor = found_samples[class_idx].unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            confidence = torch.nn.functional.softmax(output, dim=1)[0][pred_idx] * 100

        targets = [ClassifierOutputTarget(class_idx)]
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0, :]

        rgb_img = found_samples[class_idx].cpu().numpy().transpose(1, 2, 0)
        rgb_img = (rgb_img * mri_std) + mri_mean
        rgb_img = np.clip(rgb_img, 0, 1).astype(np.float32)


        grayscale_cam = gaussian_filter(grayscale_cam, sigma=1)
        grayscale_cam[:2, :] = 0
        grayscale_cam[-2:, :] = 0
        grayscale_cam[:, :2] = 0
        grayscale_cam[:, -2:] = 0

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)


        axes[0, idx].imshow(rgb_img)
        axes[0, idx].set_title(f"Actual: {classes[class_idx]}", fontsize=12, fontweight='bold')
        axes[0, idx].axis('off')


        color = "green" if pred_idx == class_idx else "red"
        axes[1, idx].imshow(cam_image)
        axes[1, idx].set_title(f"Pred: {classes[pred_idx]}\n({confidence:.1f}%)",
                              color=color, fontsize=12, fontweight='bold')
        axes[1, idx].axis('off')

    plt.suptitle("Model Attention Maps for Tumor Localization (Random Samples)", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()
