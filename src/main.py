import os
import random

import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary
from sklearn.metrics import classification_report

from .config import TRAIN_DIR, TEST_DIR, EPOCHS, PATIENCE
from .data_visualization import (
    plot_class_distribution_advanced,
    show_multiple_samples,
    show_sample_images,
    pixel_intensity_analysis,
    plot_data_split,
    plot_dataloader_class_distribution,
)
from .transforms import show_augmentation_effects
from .dataset import MRIDataset, get_loaders_with_val
from .model import AstraNetBT, get_model
from .optimizer import SAM
from .engine import train, evaluate, evaluate_loss
from .visualize_results import plot_confusion_matrix, plot_history
from .gradcam import visualize_brain_tumors_localization


def main():
    # ---------------------------------------------------------------
    # Exploratory Data Analysis
    # ---------------------------------------------------------------
    plot_class_distribution_advanced(TRAIN_DIR)
    plot_class_distribution_advanced(TEST_DIR)

    show_multiple_samples(TRAIN_DIR)
    show_sample_images(TRAIN_DIR)
    pixel_intensity_analysis(TRAIN_DIR)

    show_augmentation_effects(TRAIN_DIR)

    # ---------------------------------------------------------------
    # Dataset sanity check
    # ---------------------------------------------------------------
    test_ds = MRIDataset(TRAIN_DIR)
    img, label = test_ds[0]

    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.title("Verify: Brain should fill the frame now")
    plt.axis('off')
    plt.show()

    # ---------------------------------------------------------------
    # DataLoaders
    # ---------------------------------------------------------------
    train_loader, val_loader, test_loader, classes = get_loaders_with_val(
        TRAIN_DIR, TEST_DIR, batch_size=64, val_split=0.2
    )

    plot_data_split(train_loader, val_loader, test_loader)
    plot_dataloader_class_distribution(train_loader, classes, "Training Set Class Distribution")
    plot_dataloader_class_distribution(test_loader, classes, "Test Set Class Distribution")

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------
    model = AstraNetBT(num_classes=4)
    summary(model, input_size=(1, 3, 224, 224))

    # ---------------------------------------------------------------
    # Optimizer / criterion / scheduler
    # ---------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=4).to(device)
    base_optimizer = torch.optim.Adam
    optimizer = SAM(model.parameters(),
                    torch.optim.Adam,
                    lr=1e-4,
                    weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.3,
                                                           patience=2)

    # ---------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------
    patience_counter = 0
    best_val_loss = float("inf")

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model,
                                      train_loader,
                                      optimizer,
                                      criterion,
                                      device)
        val_loss, val_acc = evaluate(model,
                                     val_loader,
                                     criterion,
                                     device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: [T: {train_loss:.3f}, V: {val_loss:.3f}] | Acc: [T: {train_acc:.2f}%, V: {val_acc:.2f}%]")

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
            print(" -> Model Improved. Saving Weights.")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered! No improvement for {PATIENCE} epochs.")
            break
    model.load_state_dict(torch.load('best_model.pth'))

    # ---------------------------------------------------------------
    # Training history
    # ---------------------------------------------------------------
    plot_history(history)

    # ---------------------------------------------------------------
    # Model Evaluation
    # ---------------------------------------------------------------
    y_true = []
    y_pred = []
    model.eval()

    with torch.no_grad():
        for images, labels_batch in test_loader:
            images, labels_batch = images.to(device), labels_batch.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    labels = np.array(y_true)
    preds = np.array(y_pred)

    print(classification_report(labels, preds))
    plot_confusion_matrix(labels, preds, classes)

    # ---------------------------------------------------------------
    # Grad-CAM visualization
    # ---------------------------------------------------------------
    visualize_brain_tumors_localization(model, test_loader, classes, device=device)
    visualize_brain_tumors_localization(model, test_loader, classes, device=device)
    visualize_brain_tumors_localization(model, test_loader, classes, device=device)


if __name__ == "__main__":
    main()
