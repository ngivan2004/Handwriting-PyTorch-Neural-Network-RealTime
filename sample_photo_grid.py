import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

# Define transformations
tensor_transform = transforms.ToTensor()
normalize_transform = transforms.Normalize((0.1307,), (0.3081,))
augmentation_transform = transforms.Compose([
    transforms.RandomAffine(degrees=30,
                            translate=(0.2, 0.2),
                            scale=(0.25, 1),
                            shear=(-30, 30, -30, 30)),
    tensor_transform,
    normalize_transform
])

# Load dataset
training_data = datasets.MNIST(
    root="data", train=True, download=True, transform=tensor_transform)

# Function to create grid of images


def create_image_grid(images, labels, title, rows=10, cols=10):
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    fig.suptitle(title, fontsize=16)
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = images[i].squeeze()
            ax.imshow(img, cmap='gray')
            ax.set_title(f'{labels[i]}', fontsize=8)
            ax.axis('off')
    plt.tight_layout()
    plt.show()


# Get a sample of 100 images and their labels
sample_indices = torch.randint(len(training_data), size=(100,))
sample_images = [training_data[i][0] for i in sample_indices]
sample_labels = [training_data[i][1] for i in sample_indices]

# Display original images
create_image_grid(sample_images, sample_labels, "Original Images")

# Apply transformations to the sample images
augmented_images = []
for image in sample_images:
    pil_image = to_pil_image(image)  # Convert tensor to PIL Image
    # Apply augmentation and transformation
    augmented_image = augmentation_transform(pil_image)
    augmented_images.append(augmented_image)

# Display augmented images
create_image_grid(augmented_images, sample_labels, "Augmented Images")
