import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed=42):
    """
    Set random seed for reproducibility
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


def count_trainable_parameters(model):
    """
    Count trainable parameters
    """

    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total_params


def plot_training_curves(
    train_losses,
    val_losses
):
    """
    Plot training curves
    """

    plt.figure(figsize=(8, 5))

    plt.plot(
        train_losses,
        label="Train Loss"
    )

    plt.plot(
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title(
        "Model Training Curve"
    )

    plt.legend()

    plt.grid(alpha=0.3)

    plt.show()


def predict_probabilities(
    model,
    X_tensor
):
    """
    Predict probabilities
    """

    model.eval()

    with torch.no_grad():

        logits = model(
            X_tensor
        )

        probabilities = torch.sigmoid(
            logits
        )

    return probabilities