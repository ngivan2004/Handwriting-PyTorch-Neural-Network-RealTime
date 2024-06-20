import torch


from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.MNIST(root="data", train=True, download=True,
                               transform=ToTensor())
testing_data = datasets.MNIST(root="data", train=False, download=True,
                              transform=ToTensor())


figure = plt.figure(figsize=(8, 8))

# Select one random image from the training data
sample_idx = torch.randint(len(training_data), size=(1,)).item()
img, label = training_data[sample_idx]

# Add a single subplot
ax = figure.add_subplot(1, 1, 1)
plt.title(label)
plt.axis("off")
plt.imshow(img.squeeze(), cmap="gray")

# Overlay pixel values
img_data = img.squeeze().numpy()
for y in range(img_data.shape[0]):
    for x in range(img_data.shape[1]):
        pixel_value = img_data[y, x]
        ax.text(x, y, f'{pixel_value:.1f}', color='red',
                fontsize=6, ha='center', va='center')

# Show the figure
plt.show()
