import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def plot_class_distribution_advanced(data_dir):
    classes = sorted(os.listdir(data_dir))
    counts = [len(os.listdir(os.path.join(data_dir, cls))) for cls in classes]

    total = sum(counts)
    percentages = [(c / total) * 100 for c in counts]

    deep_colors = ['#4C72B0',
                   '#55A868',
                   '#C44E52',
                   '#8172B2',
                   '#CCB974',
                   '#64B5CD',
                   '#4E79A7',
                   '#F28E2B']
    bar_colors = [deep_colors[i % len(deep_colors)] for i in range(len(classes))]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes,
                      counts,
                          color=bar_colors,
                                   edgecolor='black',
                                              alpha=0.9)

    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (max(counts) * 0.01),
                 f"{counts[i]}\n({percentages[i]:.1f}%)",
                 ha='center', va='bottom', fontsize=9)

    plt.title("Class Distribution (Count + %)", fontsize=14)
    plt.xlabel("Classes")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


def show_multiple_samples(data_dir, num_samples=4):
    classes = os.listdir(data_dir)

    plt.figure(figsize=(12,8))

    for i, cls in enumerate(classes):
        imgs = os.listdir(os.path.join(data_dir, cls))[:num_samples]

        for j, img_name in enumerate(imgs):
            img = cv2.cvtColor(
                cv2.imread(os.path.join(data_dir, cls, img_name)),
                cv2.COLOR_BGR2RGB
            )

            plt.subplot(len(classes), num_samples, i*num_samples + j + 1)
            plt.imshow(img)
            plt.axis('off')
            if j == 0:
                plt.ylabel(cls)

    plt.show()


def show_sample_images(data_dir):
    classes = os.listdir(data_dir)

    plt.figure(figsize=(10,8))

    for i, cls in enumerate(classes):
        img_path = os.path.join(data_dir,
                                cls,
                                os.listdir(os.path.join(data_dir, cls))[0])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(2,2,i+1)
        plt.imshow(img)
        plt.title(cls)
        plt.axis("off")

    plt.show()


def pixel_intensity_analysis(data_dir,
                             max_images_per_class=50):
    pixels = []

    for cls in sorted(os.listdir(data_dir)):
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        img_list = os.listdir(cls_path)[:max_images_per_class]

        for img_name in img_list:
            img_path = os.path.join(cls_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                pixels.extend(img.flatten())

    pixels = np.array(pixels)

    plt.figure(figsize=(10, 6))
    plt.hist(pixels, bins=60, color='#2C3E50', edgecolor='#1A252F', alpha=0.85)

    plt.title("Global Pixel Intensity Distribution", fontsize=14, pad=15)
    plt.xlabel("Pixel Intensity (0-255)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)

    plt.xlim(0, 255)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


#Dataset Split Visualization
def plot_data_split(train_loader,
                          val_loader,
                             test_loader):
    train_size = len(train_loader.dataset)
    val_size   = len(val_loader.dataset)
    test_size  = len(test_loader.dataset)

    labels = ["Train", "Validation", "Test"]
    sizes = [train_size, val_size, test_size]
    colors = ['#4C72B0', '#55A868', '#C44E52']

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, sizes,
                   color=colors,
                   edgecolor='black',
                   alpha=0.85)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2,
                 height + (max(sizes) * 0.01),
                 f"{int(height)}",
                 ha='center', va='bottom',
                 fontweight='bold', fontsize=10)

    plt.title("Dataset Split Distribution", fontsize=14, pad=15)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.ylim(0, max(sizes) * 1.15)
    plt.tight_layout()
    plt.show()


def plot_dataloader_class_distribution(dataloader,
                                       classes,
                                       title="Class Distribution"):
    all_labels = []
    for _, labels in dataloader:
        all_labels.extend(labels.cpu().numpy())

    label_counts = Counter(all_labels)
    sorted_counts = {classes[k]: v for k, v in sorted(label_counts.items())}

    class_names = list(sorted_counts.keys())
    counts = list(sorted_counts.values())
    total = sum(counts)
    percentages = [(c / total) * 100 for c in counts]

    deep_colors = ['#4C72B0',
                   '#55A868',
                   '#C44E52',
                   '#8172B2',
                   '#CCB974',
                   '#64B5CD',

                   '#4E79A7', '#F28E2B']
    bar_colors = [deep_colors[i % len(deep_colors)] for i in range(len(class_names))]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names,
                   counts,
                   color=bar_colors,
                   edgecolor='black',
                   alpha=0.9)

    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (max(counts) * 0.01),
                 f"{counts[i]}\n({percentages[i]:.1f}%)",
                 ha='center', va='bottom', fontsize=9)

    plt.title(f"{title} (Count + %)", fontsize=14)
    plt.xlabel("Classes")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
