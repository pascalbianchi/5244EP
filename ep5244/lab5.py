import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import itertools


class FixedKernels_processing(nn.Module):
    def __init__(self, kernels, activation=None, padding = False):
        """
        Args:
            kernels (torch.Tensor or list of torch.Tensor): A single 2D tensor or a list of 2D tensors,
                each of shape [kH, kW].
            activation (nn.Module, optional): Activation function to apply after convolution. Default is None.
            padding: Default is False
        """
        super(FixedKernels_processing, self).__init__()

        # If a single kernel is provided, wrap it in a list.
        if not isinstance(kernels, list):
            kernels = [kernels]

        # Validate that each kernel is a 2D torch.Tensor.
        for i, k in enumerate(kernels):
            if not torch.is_tensor(k):
                raise ValueError(f"Kernel at index {i} is not a torch.Tensor.")
            if k.ndim != 2:
                raise ValueError(f"Kernel at index {i} must be a 2D tensor, got shape {k.shape}.")

        # Assume all kernels share the same spatial dimensions.
        kernel_size = kernels[0].shape  # (kH, kW)
        num_kernels = len(kernels)

        # Stack kernels into a weight tensor of shape [num_kernels, 1, kH, kW]
        weight = torch.stack(kernels, dim=0).unsqueeze(1)

        # Use padding to preserve spatial dimensions (assuming odd-sized kernels)
        if padding:
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        else:
            padding = 0

        # Create a convolution layer that takes a single-channel (grayscale) input
        self.conv = nn.Conv2d(in_channels=1, out_channels=num_kernels,
                              kernel_size=kernel_size, padding=padding, bias=False)
        # Set fixed weights (non-trainable)
        self.conv.weight = nn.Parameter(weight, requires_grad=False)

        self.activation = activation

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [N, C, H, W], where C is 3 (RGB) or 1 (grayscale).
        Returns:
            torch.Tensor: Output after grayscale conversion, fixed convolution, and optional activation.
        """
        if x.ndim != 4:
            raise ValueError("Input must be a 4D tensor of shape [N, C, H, W].")

        # Convert RGB images to grayscale if necessary.
        if x.shape[1] == 3:
            # Standard RGB to grayscale conversion using weighted sum.
            rgb_weights = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            x = (x * rgb_weights).sum(dim=1, keepdim=True)  # Shape: [N, 1, H, W]
        elif x.shape[1] == 1:
            pass
        else:
            raise ValueError("Input must have either 1 or 3 channels.")

        x = self.conv(x)

        if self.activation is not None:
            x = self.activation(x)

        return x

    def visualize(self, image):
        """
        For a given image, creates subplots where each row contains:
          - the raw image in grayscale,
          - the fixed kernel,
          - and the convolution result with that kernel.

        Args:
            image: A image in one of these formats:
                       - A torch.Tensor of shape [H, W, C] or [C, H, W] or [H, W] (grayscale),
                       - or a NumPy array (which will be converted to torch.Tensor).
        """
        # Convert raw_image to a torch.Tensor if needed.
        if not torch.is_tensor(image):
            image = torch.tensor(image)

        # Ensure the image has 3 dimensions: [H, W, C] or [C, H, W] or [H, W].
        if image.ndim == 2:
            # Grayscale image [H, W] -> add channel dimension -> [H, W, 1]
            image = image.unsqueeze(0)
        if image.ndim == 3:
            # If channels are in the last dimension, permute to [C, H, W]
            if image.shape[0] not in [1, 3]:
                raise ValueError("Image must have shape [C, H, W].")
        else:
            raise ValueError("Image must have 2 or 3 dimensions.")

        # Add batch dimension: [1, C, H, W]
        image = image.unsqueeze(0)

        # Convert image to grayscale as the model expects:
        if image.shape[1] == 3:
            rgb_weights = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            gray = (image * rgb_weights).sum(dim=1, keepdim=True)  # [1,1,H,W]
        elif image.shape[1] == 1:
            gray = image
        else:
            raise ValueError("Image must have 1 or 3 channels after conversion.")

        # Run the convolution layer on the grayscale image.
        with torch.no_grad():
            out = self.conv(gray)  # Shape: [1, num_kernels, H, W]

        num_kernels = self.conv.out_channels

        # Prepare the grayscale image for plotting (remove batch and channel dimensions).
        gray_img = gray[0, 0].cpu().numpy()

        fig, axes = plt.subplots(num_kernels, 3, figsize=(6, 2 * num_kernels))

        # In case there's only one kernel, make axes 2D.
        if num_kernels == 1:
            axes = axes.reshape(1, -1)

        for idx in range(num_kernels):
            # Left: grayscale image.
            axes[idx, 0].imshow(gray_img, cmap='gray')
            axes[idx, 0].set_title("Grayscale Image")
            axes[idx, 0].axis("off")

            # Middle: visualize the fixed kernel.
            kernel = self.conv.weight[idx, 0].cpu().numpy()
            axes[idx, 1].imshow(kernel, cmap='gray')
            axes[idx, 1].set_title(f"Kernel {idx}")
            axes[idx, 1].axis("off")

            # Right: result of convolution with this kernel.
            conv_result = out[0, idx].cpu().numpy()
            axes[idx, 2].imshow(conv_result, cmap='gray')
            axes[idx, 2].set_title("Convolution Result")
            axes[idx, 2].axis("off")

        plt.tight_layout()
        plt.show()










def plot_proba_from_model(model, dataset, list_idx, device):
    num_images = len(list_idx)
    labels_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']
    model.eval()

    # Create subplots: num_images rows, 2 columns
    f, ax = plt.subplots(num_images, 2, figsize=(10, 10))

    # Ensure ax is 2D even if num_images == 1
    if num_images == 1:
        ax = ax.reshape(1, -1)

    for idx, i in enumerate(list_idx):
        image, label = dataset[i]
        image = image.to(device)
        output = model(image.unsqueeze(0))
        softmax = nn.Softmax(dim=1)
        output_proba = softmax(output)

        # Retrieve the raw image (assumes dataset.data exists)
        raw_image = dataset.data[i]
        ax[idx, 0].imshow(raw_image)
        ax[idx, 0].axis('off')

        # Build dictionary of probabilities
        dictionnary = {labels_names[j]: output_proba.squeeze()[j].item() for j in range(10)}
        probabilities = list(dictionnary.values())
        ax[idx, 1].barh(range(10), probabilities, align='center', color='skyblue')
        ax[idx, 1].set_yticks(range(10))
        ax[idx, 1].set_yticklabels(labels_names)

    plt.tight_layout()
    plt.show()









def plot_confusion_matrix(cm):
    """
    Show confusion matrix.

    Parameter :
      - cm : numpy array
    """
    labels_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
    plt.figure(figsize=(7, 7))
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix for CIFAR-10")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(labels_names))
    plt.xticks(tick_marks, labels_names, rotation=45)
    plt.yticks(tick_marks, labels_names)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()
